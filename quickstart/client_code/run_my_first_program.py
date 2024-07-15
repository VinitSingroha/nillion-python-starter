from py_nillion_client import NillionClient

def run_my_program():
    client = NillionClient()
    result = client.run_nada_program("main.py")
    print(result)

if __name__ == "__main__":
    run_my_program()
