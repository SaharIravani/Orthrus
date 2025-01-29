
	
import genome_kit as gk
from genome_kit import Genome, Interval
import numpy as np
genome = Genome("gencode.v29")
interval = Interval("chr7", "+", 117120016, 117120036, genome)
genome.dna(interval)

genes = [x for x in genome.genes if x.name == 'PKP1']
transcripts = genes[0].transcripts
t = [t for t in transcripts if t.id == 'ENST00000263946.7'][0]
print(sum([len(x) for x in t.exons]))


def find_transcript(genome, transcript_id):
    """Find a transcript in a genome by transcript ID.
    
    Args:
        genome (object): The genome object containing a list of transcripts.
        transcript_id (str): The ID of the transcript to find.
        
    Returns:
        object: The transcript object, if found.
        
    Raises:
        ValueError: If no transcript with the given ID is found.
    
    Example:
        >>> # Create sample transcripts and a genome
        >>> transcript1 = 'ENST00000263946'
        >>> genome = Genome("gencode.v29")
        >>> result = find_transcript(genome, 'ENST00000335137')
        >>> print(result.id)
        <Transcript ENST00000263946.7 of PKP1>
        >>> # If transcript ID is not found
        >>> find_transcript(genome, 'ENST00000000000')
        ValueError: Transcript with ID ENST00000000000 not found.
    """
    transcripts = [x for x in genome.transcripts if x.id.split('.')[0] == transcript_id]
    if not transcripts:
        raise ValueError(f"Transcript with ID {transcript_id} not found.")

    return transcripts[0]

def find_transcript_by_gene_name(genome, gene_name):
    """Find all transcripts in a genome by gene name.
    
    Args:
        genome (object): The genome object containing a list of transcripts.
        gene_name (str): The name of the gene whose transcripts are to be found.
        
    Returns:
        list: A list of transcript objects corresponding to the given gene name.
        
    Raises:
        ValueError: If no transcripts for the given gene name are found.
    
    Example:
        >>> # Find transcripts by gene name
        >>> transcripts = find_transcript_by_gene_name(genome, 'PKP1')
        >>> print(transcripts)
        [<Transcript ENST00000367324.7 of PKP1>,
        <Transcript ENST00000263946.7 of PKP1>,
        <Transcript ENST00000352845.3 of PKP1>,
        <Transcript ENST00000475988.1 of PKP1>,
        <Transcript ENST00000477817.1 of PKP1>]        
        >>> # If gene name is not found
        >>> find_transcript_by_gene_name(genome, 'XYZ')
        ValueError: No transcripts found for gene name XYZ.
    """
    genes = [x for x in genome.genes if x.name == gene_name]
    if not genes:
        raise ValueError(f"No genes found for gene name {gene_name}.")
    if len(genes) > 1:
        print(f"Warning: More than one gene found for gene name {gene_name}.")
        print('Concatenating transcripts from all genes.')
    transcripts = []
    for gene in genes:
        transcripts += gene.transcripts
    return transcripts

def create_cds_track(t):
    """Create a track of the coding sequence of a transcript.
    Use the exons of the transcript to create a track where the first position of the codon is one.

    Args:
        t (gk.Transcript): The transcript object.
    """
    cds_intervals = t.cdss
    utr3_intervals = t.utr3s
    utr5_intervals = t.utr5s

    len_utr3 = sum([len(x) for x in utr3_intervals])
    len_utr5 = sum([len(x) for x in utr5_intervals])
    len_cds = sum([len(x) for x in cds_intervals])

    # create a track where first position of the codon is one
    cds_track = np.zeros(len_cds, dtype=int)
    # set every third position to 1
    cds_track[0::3] = 1
    # concat with zeros of utr3 and utr5
    cds_track = np.concatenate([np.zeros(len_utr5, dtype=int), cds_track, np.zeros(len_utr3, dtype=int)])
    return cds_track

def create_splice_track(t):
    """Create a track of the splice sites of a transcript.
    The track is a 1D array where the positions of the splice sites are 1.

    Args:
        t (gk.Transcript): The transcript object.
    """
    len_utr3 = sum([len(x) for x in t.utr3s])
    len_utr5 = sum([len(x) for x in t.utr5s])
    len_cds = sum([len(x) for x in t.cdss])

    len_mrna = len_utr3 + len_utr5 + len_cds
    splicing_track = np.zeros(len_mrna, dtype=int)
    cumulative_len = 0
    for exon in t.exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1:cumulative_len] = 1

    return splicing_track

# convert to one hot
def seq_to_oh(seq):
    oh = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq):
        if base == 'A':
            oh[i, 0] = 1
        elif base == 'C':
            oh[i, 1] = 1
        elif base == 'G':
            oh[i, 2] = 1
        elif base == 'T':
            oh[i, 3] = 1
    return oh

def create_one_hot_encoding(t):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.

    Args
        t (gk.Transcript): The transcript object.
    """
    seq = "".join([genome.dna(exon) for exon in t.exons])
    oh = seq_to_oh(seq)
    return oh

def create_six_track_encoding(t, channels_last=False):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.
    Concatenate the one-hot encoding with the cds track and the splice track.

    Args
        t (gk.Transcript): The transcript object.
    """
    oh = create_one_hot_encoding(t)
    cds_track = create_cds_track(t)
    splice_track = create_splice_track(t)
    six_track = np.concatenate([oh, cds_track[:, None], splice_track[:, None]], axis=1)
    if not channels_last:
        six_track = six_track.T
    return six_track


import math
from functools import partial
import os
import json
import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block
from huggingface_hub import PyTorchModelHubMixin

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mix_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mix_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(
    nn.Module,
    PyTorchModelHubMixin,
):

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        input_dim: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Linear(input_dim, d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x, inference_params=None, channel_last=False):
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        hidden_states = hidden_states

        return hidden_states

    def representation(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Get global representation of input data.

        Args:
            x: Data to embed. Has shape (B x C x L) if not channel_last.
            lengths: Unpadded length of each data input.
            channel_last: Expects input of shape (B x L x C).

        Returns:
            Global representation vector of shape (B x H).
        """
        out = self.forward(x, channel_last=channel_last)

        mean_tensor = mean_unpadded(out, lengths)
        return mean_tensor


def mean_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Take mean of tensor across second dimension without padding.

    Args:
        x: Tensor to take unpadded mean. Has shape (B x L x H).
        lengths: Tensor of unpadded lengths. Has shape (B)

    Returns:
        Mean tensor of shape (B x H).
    """
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked_tensor = x * mask.unsqueeze(-1)
    sum_tensor = masked_tensor.sum(dim=1)
    mean_tensor = sum_tensor / lengths.unsqueeze(-1).float()

    return mean_tensor


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def load_model(run_path: str, checkpoint_name: str) -> nn.Module:
    """Load trained model located at specified path.

    Args:
        run_path: Path where run data is located.
        checkpoint_name: Name of model checkpoint to load.

    Returns:
        Model with loaded weights.
    """
    model_config_path = os.path.join(run_path, "model_config.json")
    data_config_path = os.path.join(run_path, "data_config.json")

    with open(model_config_path, "r") as f:
        model_params = json.load(f)

    # TODO: Temp backwards compatibility
    if "n_tracks" not in model_params:
        with open(data_config_path, "r") as f:
            data_params = json.load(f)
        n_tracks = data_params["n_tracks"]
    else:
        n_tracks = model_params["n_tracks"]

    model_path = os.path.join(run_path, checkpoint_name)

    model = MixerModel(
        d_model=model_params["ssm_model_dim"],
        n_layer=model_params["ssm_n_layers"],
        input_dim=n_tracks
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict[k.lstrip("model")[1:]] = v

    model.load_state_dict(state_dict)
    return model



run_name="ssm_6_512_lr0.001_wd5e-05_mask0.15_seed0_splice_all_basic_eutheria_gene-dict"
checkpoint="epoch=22-step=20000.ckpt"
model_repository="/scratch/hdd001/home/phil/msk_backup/runs/"
model = load_model(f"{model_repository}{run_name}", checkpoint_name=checkpoint)
model = model.to(torch.device('cuda'))
print(model)
