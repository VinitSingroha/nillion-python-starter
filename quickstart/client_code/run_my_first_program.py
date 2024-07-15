def encrypt_decrypt_message(message, key, encrypt=True):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    shifted_alphabet = alphabet[key:] + alphabet[:key]
    if not encrypt:
        shifted_alphabet, alphabet = alphabet, shifted_alphabet
    table = str.maketrans(alphabet, shifted_alphabet)
    return message.translate(table)

def main():
    # Example usage
    original_message = "hello"
    key = 3
    encrypted_message = encrypt_decrypt_message(original_message, key, True)
    decrypted_message = encrypt_decrypt_message(encrypted_message, key, False)
    print(f"Original: {original_message}")
    print(f"Encrypted: {encrypted_message}")
    print(f"Decrypted: {decrypted_message}")

if __name__ == "__main__":
    main()
