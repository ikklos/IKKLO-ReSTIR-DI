import argparse
import array
import sys
from pathlib import Path


def read_non_comment_line(f):
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected end of file.")
        stripped = line.strip()
        if stripped.startswith(b"#") or not stripped:
            continue
        return stripped


def load_pfm(path: Path):
    with path.open("rb") as f:
        header = read_non_comment_line(f)
        if header == b"PF":
            channels = 3
        elif header == b"Pf":
            channels = 1
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dims = read_non_comment_line(f)
        width_str, height_str = dims.split()
        width = int(width_str)
        height = int(height_str)

        scale_line = read_non_comment_line(f)
        scale = float(scale_line)
        little_endian = scale < 0

        data = array.array("f")
        data.frombytes(f.read())

        expected = width * height * channels
        if len(data) != expected:
            raise ValueError(f"Expected {expected} floats, found {len(data)}.")

        if sys.byteorder == "little":
            host_little_endian = True
        else:
            host_little_endian = False

        if little_endian != host_little_endian:
            data.byteswap()

        return width, height, channels, scale, data


def save_pfm(path: Path, width: int, height: int, channels: int, scale: float, data: array.array):
    out = array.array("f", data)

    if sys.byteorder == "little":
        host_little_endian = True
    else:
        host_little_endian = False

    little_endian = scale < 0
    if little_endian != host_little_endian:
        out.byteswap()

    header = "PF" if channels == 3 else "Pf"
    with path.open("wb") as f:
        f.write(f"{header}\n".encode("ascii"))
        f.write(f"{width} {height}\n".encode("ascii"))
        f.write(f"{scale}\n".encode("ascii"))
        f.write(out.tobytes())


def flip_vertical(width: int, height: int, channels: int, data: array.array):
    row_size = width * channels
    flipped = array.array("f", [0.0] * len(data))
    for y in range(height):
        src_start = y * row_size
        src_end = src_start + row_size
        dst_start = (height - 1 - y) * row_size
        flipped[dst_start:dst_start + row_size] = data[src_start:src_end]
    return flipped


def main():
    parser = argparse.ArgumentParser(description="Flip a PFM image vertically.")
    parser.add_argument("input", type=Path, help="Input PFM path")
    parser.add_argument("output", type=Path, nargs="?", help="Output PFM path")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output if args.output else input_path.with_name(input_path.stem + "_flipped.pfm")

    width, height, channels, scale, data = load_pfm(input_path)
    flipped = flip_vertical(width, height, channels, data)
    save_pfm(output_path, width, height, channels, scale, flipped)

    print(f"Flipped PFM written to: {output_path}")


if __name__ == "__main__":
    main()
