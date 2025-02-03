import struct
from pathlib import Path

import pandas as pd


def read_binary_ply_to_dataframe(ply_filepath: Path) -> pd.DataFrame:
    assert ply_filepath.is_file(), f"File not found: {ply_filepath}"

    with open(ply_filepath, "rb") as file:
        binary_data = file.read()

    # ===== Parsing header
    header_end = binary_data.find(b"end_header\n") + len(b"end_header\n")
    header = binary_data[:header_end].decode("ascii")
    titels = []
    types = []
    for line in header.split("\n"):
        if line.startswith("property"):
            _, data_type, name = line.split()
            titels.append(name)
            types.append(data_type)

    # === Define the struct based on the properties
    format_map = {"double": "d", "uint": "I", "string8": "8s", "string32": "32s"}
    struct_format = "<" + "".join(format_map[t] for t in types)
    struct_size = struct.calcsize(struct_format)

    # ==== parse binary data struct by struct
    binary_data = binary_data[header_end:]
    data = []
    for i in range(0, len(binary_data), struct_size):
        struct_binary_data = binary_data[i : i + struct_size]
        struct_data = struct.unpack(struct_format, struct_binary_data)
        # strings comes as binary data, convert to regular strings
        struct_data = tuple(s.decode("ascii").strip("\x00") if isinstance(s, bytes) else s for s in struct_data)
        data.append(struct_data)

    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(data, columns=titels)

    return df
