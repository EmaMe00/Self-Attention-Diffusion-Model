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

def generate_images_grid(model, epoch, num_images=64, save_path="./images/", img_size=(32, 32, 3), device=device):
    os.makedirs(save_path, exist_ok=True)
    model.eval()  # Mettiamo il modello in modalità valutazione
    with torch.no_grad():
        # Generazione immagini per batch
        x = torch.randn(num_images, img_size[2], img_size[1], img_size[0]).to(device)  # Rumore casuale
        for i in range(n_steps):
            t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device)
            pred_noise = model(x.float(), t.unsqueeze(0))
            x = p_xt(x, pred_noise, t.unsqueeze(0))

        # Creazione griglia immagini
        grid = make_grid(x, nrow=8, normalize=True, value_range=(-1, 1))
        grid_path = os.path.join(save_path, f"epoch_{epoch}.png")
        save_image(grid, grid_path)  # Salviamo la griglia
    model.train()  # Torniamo in modalità training

# Assumendo che 'tensor_to_image', 'model', e 'p_xt' siano già definiti altrove
def save_generated_images(tot_img, n_steps, device, model, output_file="output.png"):
    x = torch.randn(tot_img, img_size[2], img_size[1], img_size[0]).to(device)
    ims = []

    # Generazione delle immagini
    for i in range(n_steps):
        t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device)
        with torch.no_grad():
            pred_noise = model(x.float(), t.unsqueeze(0))
            x = p_xt(x, pred_noise, t.unsqueeze(0))

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
