"""Build a portable phrase bank from an already approved public query batch."""
import argparse
import json
from pathlib import Path

from cave.lunar_query_compiler_v15 import build_bank,bank_hash,load_bank
from cave.lunar_query_families_v14 import load_protocol
from cave.ppio_client_v11 import sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--route-contract',type=Path,required=True)
    parser.add_argument('--query-protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=load_protocol(args.query_protocol)
    bank=build_bank(json.loads(args.response.read_text()),json.loads(args.route_contract.read_text()),protocol,
                    source_response_file_sha256=sha256(args.response))
    load_bank(bank,protocol)
    write_json(args.output,bank,durable=True)
    print(json.dumps(dict(passed=True,bank_sha256=bank_hash(bank),file_sha256=sha256(args.output),
                          phrase_counts={key:len(rows) for key,rows in bank['phrases'].items()},new_HTTP_requests=0)),flush=True)


if __name__=='__main__':main()
