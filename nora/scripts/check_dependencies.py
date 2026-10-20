"""Read-only package availability check; not a reproducible dependency lock."""
from importlib import metadata
import importlib.util
import json
import sys

MODULES = {
    'torch':'torch','transformers':'transformers','accelerate':'accelerate',
    'bitsandbytes':'bitsandbytes','peft':'peft','sentence_transformers':'sentence-transformers',
    'faiss':'faiss-cpu','rank_bm25':'rank-bm25','numpy':'numpy','scipy':'scipy',
    'sklearn':'scikit-learn','sentencepiece':'sentencepiece','fastapi':'fastapi',
    'uvicorn':'uvicorn','pydantic':'pydantic','yaml':'PyYAML','cryptography':'cryptography',
    'jsonschema':'jsonschema','referencing':'referencing','gradio':'gradio'
}
rows = {}
for module, package in MODULES.items():
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = None
    rows[package] = {'import_available': importlib.util.find_spec(module) is not None,'version':version}
print(json.dumps({'python':sys.version.split()[0], 'packages':rows,
    'note':'Availability only; run pip check, pinned-version review and actual inference tests before accepting the environment.'},indent=2))
