from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


class Vault:
    def __init__(self, kv_name: str):
        self.kv_uri = f"https://{kv_name}.vault.azure.net"
        credentials = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
        self.client = SecretClient(vault_url=self.kv_uri, credential=credentials)

    @lru_cache
    def get_secret(self, secret_name: str):
        """
        Retrieve keyvault from Azure
        Returns:
        str: keyvault
        """
        return self.client.get_secret(secret_name).value
