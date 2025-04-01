from myClass.package import *
from myClass.diffusion import *
from model_param import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.1)
            init.zeros_(m.bias)

def generate_images_grid(model, epoch, device, diffusion, num_images=64, save_path="./images/", img_size=(32, 32, 3)):
    os.makedirs(save_path, exist_ok=True)
    model.eval()  # Mettiamo il modello in modalità valutazione
    with torch.no_grad():
        # Generazione immagini per batch
        x = torch.randn(num_images, img_size[2], img_size[1], img_size[0]).to(device)  # Rumore casuale
        for i in range(diffusion.n_steps):
            t = torch.tensor(diffusion.n_steps - i - 1, dtype=torch.long).to(device)
            pred_noise = model(x.float(), t.unsqueeze(0))
            x = diffusion.p_xt(x, pred_noise, t.unsqueeze(0))

        # Creazione griglia immagini
        grid = make_grid(x, nrow=8, normalize=True, value_range=(0, 1))
        grid_path = os.path.join(save_path, f"epoch_{epoch}.png")
        save_image(grid, grid_path)  # Salviamo la griglia
    model.train()  # Torniamo in modalità training

# Assumendo che 'tensor_to_image', 'model', e 'p_xt' siano già definiti altrove
def save_generated_images(tot_img, n_steps, device, model, diffusion, output_file="output.png"):
    x = torch.randn(tot_img, img_size[2], img_size[1], img_size[0]).to(device)
    ims = []

    # Generazione delle immagini
    for i in range(n_steps):
        t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device)
        with torch.no_grad():
            pred_noise = model(x.float(), t.unsqueeze(0))
            x = diffusion.p_xt(x, pred_noise, t.unsqueeze(0))

    # Conversione dei tensori in immagini
    for i in range(tot_img):
        ims.append(tensor_to_image(x[i].unsqueeze(0).cpu()))

    # Creazione dell'immagine composita
    image = Image.new('RGB', size=(img_size[1] * 10, img_size[0] * 10))
    for i, im in enumerate(ims):
        image.paste(im, ((i % 10) * img_size[1], img_size[0] * (i // 10)))

    # Salvataggio in formato PNG
    resized_image = image.resize((img_size[1] * 4 * 10, img_size[0] * 4 * 10), Image.NEAREST)
    resized_image.save(output_file)


    print("\nImmagine salvata! \n")

# Pulisce le directory
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


def calculate_fid(path_fake, path_real, device, model, diffusion, full_dataset, python_version="python3"):

    j = 0
    for k in range(15):  # Genera 15 batch di 100 immagini (15 * 100 = 1500)
        x = torch.randn(100, img_size[2], img_size[1], img_size[0]).to(device)  # Inizia con rumore casuale
        print(f"Generazione batch {k + 1}")
        
        # Passo della diffusione (rimozione del rumore iterativa)
        for i in range(diffusion.n_steps):
            t = torch.tensor(diffusion.n_steps - i - 1, dtype=torch.long).to(device)  # Calcola il passo temporale
            with torch.no_grad():
                pred_noise = model(x.float(), t.unsqueeze(0))  # Prevedi il rumore
                x = diffusion.p_xt(x, pred_noise, t.unsqueeze(0))  # Rimuovi il rumore (passaggio inverso)

        for x0 in x:
            # Salva immagine generata
            save_image(x0.unsqueeze(0).to(device), os.path.join(path_fake, f"{j}.png"))
            
            # Estrai immagine reale dal dataset
            real_image, _ = full_dataset[j]  # Restituisce (immagine, etichetta)
            save_image(real_image, os.path.join(path_real, f"{j}.png"))

            j += 1

    try:
        # Esegui il comando pytorch_fid e cattura l'output
        command = f"{python_version} -m pytorch_fid --device cpu {path_fake} {path_real}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            return result.stdout.strip()  # Rimuove gli spazi vuoti extra
        else:
            print(f"Errore durante il calcolo del FID: {result.stderr}")
            return None
    except Exception as e:
        print(f"Errore durante l'esecuzione del comando FID: {str(e)}")
        return None