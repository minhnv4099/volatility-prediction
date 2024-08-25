import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Optional

torch.manual_seed(123)

@dataclass
class VolatilityPredictorConfig:

    input_dim: Optional[int] = field(
        default=6,
        metadata='Dimension of input vector.'
    )
    up_projection: Optional[int] = field(
        default=12,
        metadata="Up projection input vector."
    )

    up_projection_bias: Optional[bool] = field(
        default=True,
        metadata='Add bias when up projection.'
    )

    lstm_dim: Optional[int] = field(
        default=64,
        metadata="Dimension of hidden."
    )

    lstm_bias: Optional[bool] = field(
        default=True,
        metadata='Add bias when comput lstm.'
    )

    proj_size: Optional[int] = field(
        default=0,
        metadata='Project size of lstm.'
    )

    is_bidirectional: Optional[bool] = field(
        default=False,
        metadata="Compute back to head when conpute lsmt."
    )

    num_lstm: Optional[int] = field(
        default=2,
        metadata="Number of stacking lstm."
    )

    down_projection: Optional[int] = field(
        default=32,
        metadata='Final vector dim before compute logits.'
    )

    down_projection_bias: Optional[bool] = field(
        default=True,
        metadata='Add bias when down projection.'
    )

    final_projection: Optional[int] = field(
        default=1,
        metadata='Dimension of logits.'
    )

    final_projection_bias: Optional[bool] = field(
        default=True,
        metadata='Add bias when final projection.'
    )   

    device: Optional[str] = field(
        default='cpu',
        metadata='Device put model and data.'
    )

    dtype: Optional[object] = field(
        default=torch.float32,
        metadata="Dtype of model's parameters."
    )


class VolatilityPredictor(nn.Module):
    
    def __init__(self, config: VolatilityPredictorConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        
        self.up_projection = nn.Linear(
            in_features=self.config.input_dim,
            out_features=self.config.up_projection,
            bias=self.config.up_projection_bias,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        self.relu1 = nn.ReLU(inplace=False)

        self.lstm = nn.LSTM(
            input_size=self.config.up_projection,
            hidden_size=self.config.lstm_dim,
            num_layers=self.config.num_lstm,
            bidirectional=self.config.is_bidirectional,
            batch_first=True,
            bias=self.config.lstm_bias,
            proj_size=self.config.proj_size,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        self.lstm_dim = self.config.lstm_dim if self.config.proj_size <= 0 else self.config.proj_size

        self.down_projection = nn.Linear(
            in_features=self.lstm_dim,
            out_features=self.config.down_projection,
            bias=self.config.down_projection_bias,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        self.relu2 = nn.ReLU()

        self.final_projection = nn.Linear(
            in_features=self.config.down_projection,
            out_features=self.config.final_projection,
            bias=self.config.final_projection_bias,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def forward(self, x: torch.Tensor):
        x = self.up_projection(x)
        x = self.relu1(x)
        x, (hn, cn) = self.lstm(x[:, -1, :])
        x = self.down_projection(x)
        x = self.relu2(x)
        logits = self.final_projection(x)

        return logits