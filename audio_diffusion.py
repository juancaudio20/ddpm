import torch
import torchaudio
import argparse
import os
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler, LTPlugin
from pathlib import Path

SAMPLE_RATE = 44100
BATCH_SIZE = 1
NUM_SAMPLES = 2**18
base_dir = Path('/nas/home/jalbarracin/ddpm_1')


UNet = LTPlugin(
UNetV0, num_filters=128, window_length=64, stride=64
)

def create_model():
    return DiffusionModel(
        net_t=UNet,  # The model type used for diffusion (U-Net V0 in this case)
        in_channels=2,  # U-Net: number of input/output (audio) channels
        channels =[256, 256, 512, 512, 1024, 1024], #[8, 32, 64, 128, 256, 512, 512, 1024, 1024],   U-Net: channels at each layer
        factors =[1, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
        items =[2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
        attentions=[0, 0, 0, 0, 1, 1],  # U-Net: attention enabled/disabled at each layer
        attention_heads=12,  # U-Net: number of attention heads per attention item
        attention_features=64,  # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion,  # The diffusion method used
        sampler_t=VSampler,  # The diffusion sampler used
    )


def main():
    args = parse_args()
    dataset = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    directory = '/nas/home/jalbarracin/datasets/hrir_st'
    i = 0
    for files in os.listdir(directory):
        f = os.path.join(directory, files)
        wave, sr = torchaudio.load(f, normalize=True)
        dataset.append(wave)

    #print("data ",len(data[0][1]))
    #wave, sr = torchaudio.load(f'/nas/home/jalbarracin/datasets/hrir_st/pp{}_HRIRs_measured_{}.wav', Normalize = True)

    #dataset = WAVDataset('/nas/home/jalbarracin/datasets/hrir_st', sample_rate=44100)
    #dataset = dataset.__getitem__(dataset,1)
    #dataset = HutubsPlane(base_dir / 'HUTUBS', plane='horizontal', domain='time', side='both', download=True)


    print(f"Dataset length: {len(dataset)}")
    print(f"Sample channels and length: {dataset[0].shape}")

    torchaudio.save("test.wav", dataset[0], SAMPLE_RATE)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = create_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.run_id is not None:
        run_id = args.run_id
    print(f"Run ID: {run_id}")

    epoch = 0
    step = 0

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = f"checkpoint-{run_id}.pt"

    scaler = torch.cuda.amp.GradScaler()

    model.train()

    audio = torch.randn(1, 2, 2 ** 18).to(device)  # [batch_size, in_channels, length]
    loss = model(audio)
    loss.backward()

    # Turn noise into new audio sample with diffusion
    noise = torch.randn(1, 2, 2 ** 18).to(device)  # [batch_size, in_channels, length]
    sample = model.sample(noise, num_steps=10)  # Suggested num_steps 10-100
    sample[0].cpu()
    print(sample[0])
    torchaudio.save('test_generated_sound.wav', sample[0], sr)
'''
    while epoch < 100:
        avg_loss = 0
        avg_loss_step = 0
        progress = tqdm(dataloader)
        for i, audio in enumerate(progress):
            optimizer.zero_grad()
            audio = audio.to(device)
            with torch.cuda.amp.autocast():
                loss = model(audio)
                avg_loss += loss.item()
                avg_loss_step += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(
                loss=loss.item(),
                epoch=epoch + i / len(dataloader),
            )

            if step % 500 == 0:
                # Turn noise into new audio sample with diffusion
                noise = torch.randn(1, 2, NUM_SAMPLES, device=device)
                with torch.cuda.amp.autocast():
                    sample = model.sample(noise, num_steps=100)

                torchaudio.save(f'test_generated_sound_{step}.wav', sample[0].cpu(), SAMPLE_RATE)
                del sample
                gc.collect()
                torch.cuda.empty_cache()

            if step % 100 == 0:
                avg_loss = 0
                avg_loss_step = 0

            step += 1

        epoch += 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()