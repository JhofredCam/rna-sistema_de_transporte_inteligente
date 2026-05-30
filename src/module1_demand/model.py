import torch
import torch.nn as nn


class TransportLSTM(nn.Module):
    """
    LSTM para predicción de demanda.

    Features temporales:
        pasajeros
        dia_semana
        mes
        festivo

    Variables categóricas:
        ruta
        clima
    """

    def __init__(
        self,
        num_routes,
        num_climas,
        input_size=4,
        hidden_size=128,
        num_layers=2,
        route_embedding_dim=8,
        clima_embedding_dim=4,
        dropout=0.2,
    ):
        super().__init__()

        self.route_embedding = nn.Embedding(
            num_routes,
            route_embedding_dim,
        )

        self.clima_embedding = nn.Embedding(
            num_climas,
            clima_embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        fc_input_size = (
            hidden_size
            + route_embedding_dim
            + clima_embedding_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x,
        route_ids,
        clima_ids,
    ):
        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out[:, -1, :]

        route_embed = self.route_embedding(route_ids)

        clima_embed = self.clima_embedding(clima_ids)

        combined = torch.cat([lstm_out,route_embed,clima_embed,],dim=1,)

        return self.fc(combined)
