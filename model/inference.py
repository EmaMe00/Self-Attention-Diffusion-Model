from myClass.package import *
from myClass.usefull_func import *
from myClass.unet import *

dataset = 'CIFAR10'
n_steps = 1000 
tot_img = 100 # MAX 100
metrics = False
weights = "./result_CNN/unit.pt"
py_vers = 'python3.11'

if dataset == 'MNIST':
    transform = Compose([
        ToTensor(), 
        Pad(2),
    ])
elif dataset == 'CIFAR10':
    transform = Compose([
        ToTensor(), 
    ])
elif dataset == 'CELEBA':
    transform = Compose([
        ToTensor(),
        Resize((256, 256)),
    ])

# Load CIFAR10 or MNIST dataset based on the selected option
if dataset == 'CIFAR10':
    train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
    test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
elif dataset == 'MNIST':
    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
elif dataset == 'CELEBA':
    train_dataset = ImageFolder(root=dataset_path + "/celebA/", transform=transform)
    full_dataset = train_dataset

data_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=True)

model_config = {
    'im_channels': img_size[2],            # RGB image input
    'down_channels': [64, 128, 256, 256],  # Number of channels in downsampling layers
    'mid_channels': [256, 256, 256],       # Midblock channels
    'time_emb_dim': 256,                   # Time embedding dimension
    'down_sample': [True,True,False],      # Whether to downsample at each layer
    'num_down_layers': 2,                  # Number of layers in downblock
    'num_mid_layers': 2,                   # Number of layers in midblock
    'num_up_layers': 2,                    # Number of layers in upblock
    'num_attention_blocks':0               # Number of attention in every block
}

model = Unet(model_config).to(device)
model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
model.eval()

save_generated_images(tot_img=tot_img, n_steps=n_steps, device=device, model=model, output_file="generated_images.png")

# Rimuove il contenuto delle cartelle usate per il FID
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for file_or_dir in os.listdir(directory_path):
            file_or_dir_path = os.path.join(directory_path, file_or_dir)
            if os.path.isfile(file_or_dir_path):
                os.remove(file_or_dir_path)
            elif os.path.isdir(file_or_dir_path):
                shutil.rmtree(file_or_dir_path)
    else:
        print(f"La cartella '{directory_path}' non esiste.")

if metrics == True:

    output_fake_dir = "./content/fake/"
    output_real_dir = "./content/real/"
    os.makedirs(output_fake_dir, exist_ok=True)
    os.makedirs(output_real_dir, exist_ok=True)

    trans = transforms.ToTensor()
    j = 0
    for k in range(1):  # Genera 5 batch di tot_img immagini (5 * tot_img)
        x = torch.randn(tot_img, 3, 32, 32).to(device)  # Inizia con rumore casuale
        print(f"Generazione batch {k + 1}")
        for i in range(n_steps):
            t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device)
            with torch.no_grad():
                pred_noise = model(x.float(), t.unsqueeze(0))
                x = p_xt(x, pred_noise, t.unsqueeze(0))
        for x0 in x:
            # Salva immagine generata
            save_image(x0.unsqueeze(0).cpu(), os.path.join(output_fake_dir, f"{j}.png"))

            # Estrai immagine reale dal dataset unito
            real_image, _ = full_dataset[j]  # Restituisce (immagine, etichetta)
            save_image(real_image, os.path.join(output_real_dir, f"{j}.png"))

            j += 1

    command = py_vers +  ' -m pytorch_fid --device cpu "./content/fake" "./content/real"'
    os.system(command)

    # Pulisce entrambe le directory
    clear_directory(output_fake_dir)
    clear_directory(output_real_dir)
