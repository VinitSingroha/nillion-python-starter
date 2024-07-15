"""
In this example, we:
1. connect to the local nillion-devnet
2. generate dummy data for linear regression
3. store the linear regression program (simulated as a string)
4. store training data as a secret to be used in the computation
5. compute the linear regression model with the stored training data and new input data
"""

import asyncio
import py_nillion_client as nillion
import os
import numpy as np

from py_nillion_client import NodeKey, UserKey
from dotenv import load_dotenv
from nillion_python_helpers import get_quote_and_pay, create_nillion_client, create_payments_config

from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")

async def main():
    # 1. Initial setup
    # 1.1. Get cluster_id, grpc_endpoint, & chain_id from the .env file
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    # 1.2 pick a seed and generate user and node keys
    seed = "my_seed"
    userkey = UserKey.from_seed(seed)
    nodekey = NodeKey.from_seed(seed)

    # 2. Initialize NillionClient against nillion-devnet
    # Create Nillion Client for user
    client = create_nillion_client(userkey, nodekey)

    party_id = client.party_id
    user_id = client.user_id

    # 3. Generate dummy data for linear regression
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    X = X.flatten().tolist()
    y = y.flatten().tolist()

    # 4. Simulate storing the linear regression program as a string (instead of using a path)
    program_name = "linear_regression"
    program_code = """
    def linear_regression(X, y, X_new):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model.predict(X_new)
    """

    # Assume that the program_code is compiled and stored
    program_mir_path = "/path/to/compiled/linear_regression.nada.bin"  # Simulated path

    # Create payments config, client and wallet
    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    # Pay to store the program and obtain a receipt of the payment
    receipt_store_program = await get_quote_and_pay(
        client,
        nillion.Operation.store_program(program_mir_path),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Store the program
    action_id = await client.store_program(
        cluster_id, program_name, program_mir_path, receipt_store_program
    )

    # Create a variable for the program_id, which is the {user_id}/{program_name}. We will need this later
    program_id = f"{user_id}/{program_name}"
    print("Stored program. action_id:", action_id)
    print("Stored program_id:", program_id)

    # 5. Create the training data secret, add permissions, pay for and store it in the network
    # Create training data
    training_data = nillion.NadaValues(
        {
            "X": nillion.SecretArray(X),
            "y": nillion.SecretArray(y),
        }
    )

    # Set the input party for the secret
    party_name = "TrainingParty"

    # Set permissions for the client to compute on the program
    permissions = nillion.Permissions.default_for_user(client.user_id)
    permissions.add_compute_permissions({client.user_id: {program_id}})

    # Pay for and store the secret in the network and print the returned store_id
    receipt_store = await get_quote_and_pay(
        client,
        nillion.Operation.store_values(training_data, ttl_days=5),
        payments_wallet,
        payments_client,
        cluster_id,
    )
    # Store a secret
    store_id = await client.store_values(
        cluster_id, training_data, permissions, receipt_store
    )
    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {store_id}")

    # 6. Create compute bindings to set input and output parties, add computation time input data and pay for & run the computation
    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party(party_name, party_id)
    compute_bindings.add_output_party(party_name, party_id)

    # Add new input data at computation time, ex: {'X_new': [6, 7]}
    computation_time_data = nillion.NadaValues({"X_new": nillion.SecretArray([6, 7])})

    # Pay for the compute
    receipt_compute = await get_quote_and_pay(
        client,
        nillion.Operation.compute(program_id, computation_time_data),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Compute on the secret
    compute_id = await client.compute(
        cluster_id,
        compute_bindings,
        [store_id],
        computation_time_data,
        receipt_compute,
    )

    # 7. Return the computation result
    print(f"The computation was sent to the network. compute_id: {compute_id}")
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event.uuid}")
            print(f"🖥️  The result is {compute_event.result.value}")
            return compute_event.result.value


if __name__ == "__main__":
    asyncio.run(main())
