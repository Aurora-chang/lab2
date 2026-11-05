"""Build a private Colab upload ZIP with code, configs, notebook, and local data."""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
archive=ROOT/'outputs/Lab2_colab.zip'
archive.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
    for name in ['README.md','requirements.txt','configs','notebooks','scripts','tests','lab2_EEG_classification','results','docs']:
        path=ROOT/name
        for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if file.is_file() and '__pycache__' not in file.parts and file.name!='.DS_Store' and file.suffix!='.pyc':
                output.write(file,file.relative_to(ROOT))
print(archive)
