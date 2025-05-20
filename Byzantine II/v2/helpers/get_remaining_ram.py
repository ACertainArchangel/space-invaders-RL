import psutil
def get_remaining_ram_in_gb():
    """Get the available memory in gigabytes"""
    available_memory = psutil.virtual_memory().available
    return available_memory / (1024 ** 3)