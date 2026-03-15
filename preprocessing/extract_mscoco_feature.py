import torch
import sys
import os

# 添加 REPA 所在的目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
# import libs.autoencoder
from PIL import Image

import libs.clip
from dataset import MSCOCODatabase
import argparse
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from typing import Callable, Optional, Tuple, Union
from pathlib import Path

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]
def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        ### here
        # if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
        #     raise click.ClickException('--dest folder must be empty')
        # os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None
def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MSCOCODatabase(root='.../train2014',
                             annFile='.../annotations/captions_train2014.json',
                             size=resolution)
        save_dir = f'data/coco{resolution}_features/train_new'
    elif args.split == "val":
        datas = MSCOCODatabase(root='.../val2014',
                             annFile='.../annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'data/coco{resolution}_features/val_test'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    from preprocessing.encoders import StabilityVAEEncoder
    model_url = '.../sd-vae-ft-ema'
    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=1)
    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)
    archive_root_dir, save_bytes, close_dest = open_dest('.../coco256_features/val')

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            # if idx !=1488:
            #     continue
            x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = vae(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            # moments = vae.encode(x).latent_dist
            # mean = moments.mean
            # std = moments.std
            # moments = torch.cat([mean,std], 1).squeeze(0)
            # moments = moments.detach().cpu().numpy()

            import io

            f = io.BytesIO()
            np.save(f, moments)
            save_bytes(os.path.join(save_dir, f'{idx}.npy'), f.getvalue())
            # np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            # img = Image.fromarray(x)
            # img.save(os.path.join(save_dir, f'{idx}.png'))
            # print(torch.min(x))
            img_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_np, mode='RGB')
            img.save(os.path.join(save_dir, f'{idx}.png'))

            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)

            cap_path = os.path.join(save_dir, f"{idx}.txt")

            if torch.is_tensor(captions):
                captions_to_save = captions
                try:
                    captions_to_save = captions_to_save.detach().cpu()
                except Exception:
                    pass
                if captions_to_save.ndim == 0:
                    captions_str = str(captions_to_save.item())
                else:
                    captions_str = np.array(captions_to_save).tolist().__repr__()
            elif isinstance(captions, (list, tuple)):
                captions_str = "\n".join([str(c) for c in captions])
            else:
                captions_str = str(captions)

            with open(cap_path, "w", encoding="utf-8") as ftxt:
                ftxt.write(captions_str)


if __name__ == '__main__':
    main()