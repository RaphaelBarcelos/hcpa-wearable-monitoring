import flwr as fl

print("[Servidor] Iniciando servidor Flower...")

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=1)
)
