from myClass.package import *
from myClass.unet import *
from myClass.diffusion import *
from model_param import *
from myClass.usefull_func import *

# Verifica se esistono pesi salvati
weights_path = './unit.pt'
start_epoch = 1  

# Load Dataset with transformations (Converting images to tensors)
# Imposta il transform con il padding solo per MNIST
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

# Creazione del modello
model = Unet(model_config, device)
for param in model.parameters():
    param.data = param.data.to(device)

# Caricamento dei pesi se esistono
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"Pesi caricati da {weights_path}. Riprendo il training.")
    # Imposta l'epoca di inizio come quella successiva a quella salvata
    try:
        start_epoch = int(torch.load(weights_path + '.epoch')) + 1
    except:
        start_epoch = 1
else:
    print(f"Non esiste un checkpoint, inizio dall'epoca 0.")

losses = [] # Store losses for later plotting

optim = torch.optim.Adam(model.parameters(), lr=lr) # Optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs, eta_min=1e-6)

print("Number of parameters: " + str(count_parameters(model)))

best_loss = float('inf')  # Inizializziamo la migliore perdita come infinita.
patience_counter = 0  # Contatore delle epoche senza miglioramenti.

losses = []  # Lista per i singoli valori di loss.
epoch_losses = []  # Lista per la perdita media per epoca.

start_time = time.time()

for epoch in tqdm(range(start_epoch, epochs + 1)):
    average_train_loss = 0  # Azzeriamo la perdita cumulativa per l'epoca
    loop_train = tqdm(enumerate(data_loader, 1), total=len(data_loader), desc="Train", position=0, leave=True)
    
    for index, (x0, label) in loop_train:
        x0 = x0.to(device)
        optim.zero_grad()
        t = torch.randint(0, n_steps, (x0.shape[0],), dtype=torch.long).to(device)
        xt, noise = q_xt_x0(x0, t)
        pred_noise = model(xt.float(), t)
        loss = F.mse_loss(noise.float(), pred_noise)
        losses.append(loss.item())  # Salviamo il valore di perdita per ogni batch
        average_train_loss += loss.item()  # Sommiamo la perdita per l'epoca

        loss.backward()
        optim.step()

        loop_train.set_description(f"Train - iteration : {epoch}")
        loop_train.set_postfix(
            avg_train_loss="{:.4f}".format(average_train_loss / index),
            refresh=True,
        )

    # Calcoliamo la perdita media per l'epoca
    epoch_loss = average_train_loss / len(data_loader)
    epoch_losses.append(epoch_loss)  # Salviamo la perdita media per l'epoca

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")
    
    # Controlliamo se la perdita è migliorata
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "./unit.pt")
        torch.save(epoch, './unit.pt.epoch')  # Salva l'epoca corrente
        print(f"Epoch {epoch}: Miglioramento trovato. Nuovo miglior modello salvato con perdita = {epoch_loss:.4f}\n")
        patience_counter = 0  # Reset del contatore della pazienza
    else:
        patience_counter += 1
        print(f"Epoch {epoch}: Nessun miglioramento. La perdita è {epoch_loss:.4f}")
        print(f"Sono {patience_counter} epoche senza miglioramento.\n")
        
    # Ogni 10 epoche, generiamo una griglia di immagini
    if (epoch-1) % 10 == 0:
        print(f"Generazione batch di {inference_batch_size} immagini\n")
        generate_images_grid(model, epoch, num_images=inference_batch_size, device=device, img_size=img_size)
        
    # Se il contatore della pazienza ha superato il limite, fermiamo il training
    if patience_counter >= patience:
        print(f"Early stopping: Nessun miglioramento per {patience} epoche consecutive. Fermiamo il training.\n")
        break
        
        
end_time = time.time()
training_time = end_time - start_time 

# Salvataggio dei valori in file CSV al termine dell'allenamento
# Salviamo losses (perdita per ogni batch)
with open('./result/losses.csv', 'w', newline='') as f_loss:
    writer = csv.writer(f_loss)
    writer.writerow(['Loss'])  # Intestazione del file CSV
    for loss_value in losses:
        writer.writerow([loss_value])

# Salviamo epoch_losses (perdita media per epoca)
with open('./result/epoch_losses.csv', 'w', newline='') as f_epoch_loss:
    writer = csv.writer(f_epoch_loss)
    writer.writerow(['Epoch', 'Loss'])  # Intestazione con epoca e perdita
    for epoch_idx, epoch_loss_value in enumerate(epoch_losses, 1):
        writer.writerow([epoch_idx, epoch_loss_value])

# Generazione del grafico per la perdita per ogni batch
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Batch Loss')
plt.title('Perdita per Batch')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./result/batch_losses.png')  # Salva il grafico in formato PNG
plt.close()

# Generazione del grafico per la perdita media per epoca
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Epoch Loss')
plt.title('Perdita Media per Epoca')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./result/epoch_losses.png')  # Salva il grafico in formato PNG
plt.close()
        
print("\n I file CSV e le immagini sono stati salvati.")
print("\nTRAINING COMPLETATO")

#Inference
if inference:
    print("\nINIZIO PROCESSO DI CALCOLO DELLE METRICHE\n")

    output_fake_dir = "./content/fake/"
    output_real_dir = "./content/real/"
    os.makedirs(output_fake_dir, exist_ok=True)
    os.makedirs(output_real_dir, exist_ok=True)

    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load("./unit.pt", weights_only=True))
    model.eval()

    # generate 1500 samples
    j = 0
    for k in range(15):  # Genera 15 batch di 100 immagini (15 * 100 = 1500)
        x = torch.randn(100, img_size[2], img_size[1], img_size[0]).to(device)  # Inizia con rumore casuale
        print(f"Generazione batch {k + 1}")
        for i in range(n_steps):
            t = torch.tensor(n_steps - i - 1, dtype=torch.long).to(device)
            with torch.no_grad():
                pred_noise = model(x.float(), t.unsqueeze(0))
                x = p_xt(x, pred_noise, t.unsqueeze(0))
        for x0 in x:
            # Salva immagine generata
            save_image(x0.unsqueeze(0).to(device), os.path.join(output_fake_dir, f"{j}.png"))
            # Estrai immagine reale dal dataset unito
            real_image, _ = full_dataset[j]  # Restituisce (immagine, etichetta)
            save_image(real_image, os.path.join(output_real_dir, f"{j}.png"))

            j += 1



    # Specifica i percorsi delle cartelle contenenti le immagini
    path_fake = './content/fake/'
    path_real = './content/real/'

    print("\nCALCOLO IL FID")

    # Funzione per eseguire il comando e catturare l'output
    def calculate_fid(path_fake, path_real, pythonVersion):
        try:
            # Esegui il comando pytorch_fid e cattura l'output
            command = f"{pythonVersion} -m pytorch_fid --device cpu {path_fake} {path_real}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Se il comando è stato eseguito correttamente, prendi l'output
            if result.returncode == 0:
                return result.stdout.strip()  # Rimuove gli spazi vuoti extra dall'output
            else:
                print(f"Errore durante il calcolo del FID: {result.stderr}")
                return None
        except Exception as e:
            print(f"Errore durante l'esecuzione del comando FID: {str(e)}")
            return None

    # Calcola il FID
    fid_output = calculate_fid(path_fake, path_real, pythonVersion)

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

    # Pulisce entrambe le directory
    clear_directory(path_fake)
    clear_directory(path_real)

    # Salvataggio del resoconto
    report_path = "./result/training_report.txt"
    with open(report_path, "w") as report_file:
        report_file.write("Training Report\n")
        report_file.write("================\n")
        report_file.write(f"Tempo totale di training: {training_time:.2f} secondi\n")
        report_file.write(f"Numero di epoche completate: {len(epoch_losses)}\n")
        report_file.write(f"Best Loss: {best_loss:.4f}\n")
        
        if fid_output:
            report_file.write(f"FID: {fid_output}\n")
        else:
            report_file.write("FID: Errore nel calcolo del FID\n")

        report_file.write("\nConfigurazione del Modello:\n")
        for key, value in model_config.items():
            report_file.write(f"{key}: {value}\n")
        
        report_file.write("\nIperparametri:\n")
        report_file.write(f"- Batch size: {data_loader.batch_size}\n")
        report_file.write(f"- Learning rate: {lr}\n")
        report_file.write(f"- Numero di passi diffusivi: {n_steps}\n")
        report_file.write(f"- Pazienza (early stopping): {patience}\n")
        report_file.write("\n")
        report_file.write("Altre statistiche:\n")
        report_file.write(f"- Perdita media per epoca salvata in epoch_losses.csv\n")
        report_file.write(f"- Perdita per batch salvata in losses.csv\n")
        
    print(f"\nResoconto del training salvato in: {report_path}")