# Importing Flower
import flwr as fl

# Importing the model and data functions
from model import create_model, load_data, partition_data

# Defining the hospital numbers
num_hospitals = int(input("Enter the number of hospitals: "))

# Defining the data, partitions and the model
(x_train, y_train), (x_test, y_test) = load_data()
partitions = partition_data(x_train, y_train, num_hospitals)
global_model = create_model()

# Method to run in each hospital
def client(cid: str):
    
    # Defining the id, local data and model for each hospital
    client_id = int(cid)
    x_local, y_local = partitions[client_id]
    model = create_model()

    # Defining the client class for each hospital
    class HospitalClient(fl.client.NumPyClient):

        def get_parameters(self, config):

            return model.get_weights()

        def fit(self, parameters, config):

            model.set_weights(parameters)
            model.fit(x_local, y_local, epochs=7, batch_size=32, verbose=0)
            return model.get_weights(), len(x_local), {}

        def evaluate(self, parameters, config):

            return 0.0, len(x_local), {}

    return HospitalClient()

# Method to evaluate the global model after each round
def evaluate(server_round, parameters, config):

    global_model.set_weights(parameters)
    loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

    print(f"\n Round {server_round}")
    print(f"Loss global: {loss}")
    print(f"Accuracy global: {accuracy}\n")

    return loss, {"accuracy": accuracy}

# Defining the strategy for the federated learning process
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    min_fit_clients=2,
    evaluate_fn=evaluate
)

# Starting the federated learning simulation
fl.simulation.start_simulation(
    client_fn=client,
    num_clients=num_hospitals,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

# Saving the final model
global_model.save("models/federated_model.keras")