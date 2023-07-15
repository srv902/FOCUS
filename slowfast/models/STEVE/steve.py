"""
STEVE model and utils to operate on video clips with image reconstruction as the primary tasks!
"""

from .utils import *
from .dvae import dVAE
from .transformer import TransformerEncoder, TransformerDecoder
from ..build import MODEL_REGISTRY
import torch.nn.functional as F

class SlotAttentionVideo(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)

                # use MLP only when more than one iterations
                if i < self.num_iterations - 1:
                    slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

            # predictor
            slots = self.predictor(slots)

        attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size

        return slots_collect, attns_collect


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, offset=0):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, offset:offset + T])


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        # print("size of inputs >> ", inputs.shape)
        # print("size of the proj pe >> ", self.projection(self.pe).shape)
        # exit()
        return inputs + self.projection(self.pe)

class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class BaseCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fenc = nn.Sequential(
            Conv2dBlock(args.SLOTS.IMG_CHANNELS, args.SLOTS.CNN_HID_SIZE, 5, 1 if args.SLOTS.IMG_SIZE == 64 else 2, 2),
            Conv2dBlock(args.SLOTS.CNN_HID_SIZE, args.SLOTS.CNN_HID_SIZE, 5, 1, 2),
            Conv2dBlock(args.SLOTS.CNN_HID_SIZE, args.SLOTS.CNN_HID_SIZE, 5, 1, 2),
            conv2d(args.SLOTS.CNN_HID_SIZE, args.SLOTS.DECODER.DIM, 5, 1, 2),
        )

    def forward(self, x):
        return self.fenc(x)

class Res18Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        from torchvision.models import resnet18
        self.res18 = resnet18()

        # change the first conv in resnet18 ..
        self.res18.conv1 = nn.Conv2d(args.SLOTS.IMG_CHANNELS, args.SLOTS.CNN_HID_SIZE, 3, 1, 1)

        # get the first two blocks of resnet18
        self.fenc = nn.Sequential(*list(self.res18.children())[:-5])

        # add upconvolution to map to image size ..
        self.upconv = nn.ConvTranspose2d(args.SLOTS.CNN_HID_SIZE, 
                                args.SLOTS.DECODER.DIM, 
                                3, 
                                stride=2, 
                                padding=1, 
                                dilation=1, 
                                output_padding=1
                        )

    def forward(self, x):
        x = self.fenc(x)
        x = F.relu(x)
        x = self.upconv(x)

        return x

def fetch_visual_encoder(args):
    if args.MODEL.CNN_NAME == 'base':
        return BaseCNN(args)
    elif args.MODEL.CNN_NAME == 'res18':
        return Res18Block(args)
    else:
        raise ValueError("Incorrect cnn name provided!")


class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # visual encoder ..
        self.cnn = fetch_visual_encoder(args)

        self.pos = CartesianPositionalEmbedding(args.SLOTS.DECODER.DIM, args.SLOTS.IMG_SIZE if args.SLOTS.IMG_SIZE == 64 else args.SLOTS.IMG_SIZE // 2)

        self.layer_norm = nn.LayerNorm(args.SLOTS.DECODER.DIM)

        self.mlp = nn.Sequential(
            linear(args.SLOTS.DECODER.DIM, args.SLOTS.DECODER.DIM, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.SLOTS.DECODER.DIM, args.SLOTS.DECODER.DIM))

        self.savi = SlotAttentionVideo(
            args.SLOTS.NUM_ITERS, args.SLOTS.NUM_SLOTS,
            args.SLOTS.DIM, args.SLOTS.SIZE, args.SLOTS.MLP_HID_SIZE,
            args.SLOTS.NUM_PREDICTOR_BLOCKS, args.SLOTS.NUM_PREDICTOR_HEADS, args.SLOTS.PREDICTOR_DROPOUT)

        self.slot_proj = linear(args.SLOTS.SIZE, args.SLOTS.DIM, bias=False)


class STEVEDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dict = OneHotDictionary(args.SLOTS.VOCAB_SIZE, args.SLOTS.DECODER.DIM)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.SLOTS.DECODER.DIM))
        nn.init.xavier_uniform_(self.bos)

        self.pos = LearnedPositionalEmbedding1D(1 + (args.SLOTS.IMG_SIZE // 4) ** 2, args.SLOTS.DECODER.DIM)

        self.tf = TransformerDecoder(
            args.SLOTS.DECODER.NUM_BLOCKS, (args.SLOTS.IMG_SIZE // 4) ** 2, args.SLOTS.DECODER.DIM, args.SLOTS.DECODER.NUM_HEADS, args.SLOTS.DECODER.DROPOUT)

        self.head = linear(args.SLOTS.DECODER.DIM, args.SLOTS.VOCAB_SIZE, bias=False)

@MODEL_REGISTRY.register()
class STEVE(nn.Module):
    """
    STEVE module working on video clips with slots to capture object related features
    """
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.SLOTS.NUM_ITERS
        self.num_slots = args.SLOTS.NUM_SLOTS
        self.cnn_hidden_size = args.SLOTS.CNN_HID_SIZE
        self.slot_size = args.SLOTS.SIZE
        self.mlp_hidden_size = args.SLOTS.MLP_HID_SIZE
        self.img_channels = args.SLOTS.IMG_CHANNELS
        self.image_size = args.SLOTS.IMG_SIZE
        self.vocab_size = args.SLOTS.VOCAB_SIZE
        self.d_model = args.SLOTS.DECODER.DIM

        # dvae
        self.dvae = dVAE(args.SLOTS.VOCAB_SIZE, args.SLOTS.IMG_CHANNELS)

        # encoder networks
        self.steve_encoder = STEVEEncoder(args)

        # decoder networks
        self.steve_decoder = STEVEDecoder(args)

    def forward(self, video, tau, hard):
        B, T, C, H, W = video.size()

        # print("video size as input >> ", video.shape)
        # exit()

        video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)       # B * T, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                  # B * T, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()         # B * T, vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                         # B * T, H_enc * W_enc, vocab_size
        z_emb = self.steve_decoder.dict(z_hard)                                                     # B * T, H_enc * W_enc, d_model
        z_emb = torch.cat([self.steve_decoder.bos.expand(B * T, -1, -1), z_emb], dim=1)             # B * T, 1 + H_enc * W_enc, d_model
        z_emb = self.steve_decoder.pos(z_emb)                                                       # B * T, 1 + H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.dvae.decoder(z_soft).reshape(B, T, C, H, W)               # B, T, C, H, W
        dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * T)                      # 1

        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        # print("size of the embedding from cnn >> ", emb.shape)

        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
                                                                # attns: B, T, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)          # B, T, num_slots, 1, H, W
        attns = video.unsqueeze(2) * attns + (1. - attns)                               # B, T, num_slots, C, H, W

        # decode
        slots = self.steve_encoder.slot_proj(slots)                                                         # B, T, num_slots, d_model
        pred = self.steve_decoder.tf(z_emb[:, :-1], slots.flatten(end_dim=1))                               # B * T, H_enc * W_enc, d_model
        pred = self.steve_decoder.head(pred)                                                                # B * T, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / (B * T)                         # 1

        return (dvae_recon.clamp(0., 1.),
                cross_entropy,
                dvae_mse,
                attns)

    def encode(self, video):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)

        # savi
        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)     # slots: B, T, num_slots, slot_size
                                                            # attns: B, T, num_slots, num_inputs

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)                      # B, T, num_slots, 1, H, W

        attns_vis = video.unsqueeze(2) * attns + (1. - attns)           # B, T, num_slots, C, H, W

        return slots, attns_vis, attns

    def decode(self, slots):
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        slots = self.steve_encoder.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.steve_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.steve_decoder.tf(
                self.steve_decoder.pos(input),
                slots
            )
            z_next = F.one_hot(self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.steve_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, video):
        """
        image: batch_size x img_channels x H x W
        """
        B, T, C, H, W = video.size()
        slots, attns, _ = self.encode(video)
        recon_transformer = self.decode(slots.flatten(end_dim=1))
        recon_transformer = recon_transformer.reshape(B, T, C, H, W)

        return recon_transformer
