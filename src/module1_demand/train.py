import torch
from data_generator import generate_data
from data_loader import TransportDataset, build_sequences, create_dataloader
from model import TransportLSTM
from preprocessor import preprocess_pipeline
from trainer import train_model

DATA_PATH = "data/demanda_transporte.csv"
SEQ_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

if __name__ == "__main__":
    print("Generando datos sintéticos...")
    generate_data(DATA_PATH, web_output_csv="web/public/data/demanda_transporte.csv")

    print("Preprocesando...")
    pipe = preprocess_pipeline(DATA_PATH, save=True, max_year=2026, output_dir="api/models/demand")

    print("Construyendo secuencias...")
    X_train, routes_train, climas_train, y_train = build_sequences(
        pipe["train_df"], pipe["feature_cols"], seq_length=SEQ_LENGTH
    )
    X_test, routes_test, climas_test, y_test = build_sequences(
        pipe["test_df"], pipe["feature_cols"], seq_length=SEQ_LENGTH
    )

    train_loader = create_dataloader(
        TransportDataset(X_train, routes_train, climas_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = create_dataloader(
        TransportDataset(X_test, routes_test, climas_test, y_test),
        batch_size=BATCH_SIZE
    )

    model = TransportLSTM(
        num_routes=len(pipe["route_encoder"].classes_),
        num_climas=len(pipe["clima_encoder"].classes_),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    history = train_model(
        model, train_loader, test_loader, device,
        epochs=EPOCHS, lr=LR,
        output_dir="api/models/demand"
    )

    print("Entrenamiento completado.")
