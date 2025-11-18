import click
import os
import requests
import tqdm
import zipfile


@click.command()
@click.argument('dataset')
def convert(dataset):
    # Create lib directory if it doesn't exist
    os.makedirs('lib', exist_ok=True)

    if dataset == 'ml-100k':
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        zip_path = 'lib/ml-100k.zip'
        if not os.path.exists(zip_path):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm.tqdm(
                desc='Downloading',
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        if not os.path.exists('lib/ml-100k'):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('lib')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


if __name__ == '__main__':
    convert()
