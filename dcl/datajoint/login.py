import os

import datajoint as dj
from dotenv import find_dotenv
from dotenv import load_dotenv


def connect_to_database():
    env_file = find_dotenv()
    print("Using .env file:", env_file)
    load_dotenv(
        env_file,
        verbose=True,
        override=True,
    )  # Load environment variables from a .env file
    print("DATAJOINT_DB_HOST:", os.getenv("DATAJOINT_DB_HOST"))
    print("DATAJOINT_DB_USER:", os.getenv("DATAJOINT_DB_USER"))
    print("DATAJOINT_SCHEMA_NAME:", os.getenv("DATAJOINT_SCHEMA_NAME"))
    host_port = os.getenv("DATAJOINT_DB_HOST")
    host, port = host_port.split(":")
    dj.config["database.host"] = host
    dj.config["database.port"] = int(port)
    dj.config["database.user"] = os.getenv("DATAJOINT_DB_USER")
    dj.config["database.password"] = os.getenv("DATAJOINT_DB_PASSWORD")
    dj.config["enable_python_native_blobs"] = True
    dj.conn()


if __name__ == '__main__':
    connect_to_database()
