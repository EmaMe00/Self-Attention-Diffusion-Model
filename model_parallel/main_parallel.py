from myClass.package import *
from myClass.unet import *
from myClass.diffusion import *
from model_param import *
from myClass.usefull_func import *

class DiffusionLightningModule(pl.LightningModule):
    def __init__(self, model_config, dataset, dataset_path, img_size, batch_size, lr, patience, save_image_epoch, inference_batch_size):
        super().__init__()

        self.model_config = model_config
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.save_image_epoch = save_image_epoch
        self.inference_batch_size = inference_batch_size

        #Da inizializzare in setup poichè qui il device non è corretto
        self.model = None
        self.diffusion = None 

        self.transform = self._get_transform()
        self.best_loss = float("inf")
        self.epoch_losses = []
        self.losses = []

    def _get_transform(self):
        if self.dataset == 'MNIST':
            return Compose([ToTensor(), Pad(2)])
        elif self.dataset == 'CIFAR10':
            return Compose([ToTensor()])
        elif self.dataset == 'CELEBA':
            return Compose([ToTensor(), Resize((self.img_size[0], self.img_size[1]))])
        elif self.dataset == 'CARS':
            return Compose([ToTensor(), Resize((self.img_size[0], self.img_size[1]))])
        return None

    def prepare_data(self):
        if self.dataset == 'CIFAR10':
            CIFAR10(self.dataset_path, train=True, download=True)
            CIFAR10(self.dataset_path, train=False, download=True)
        elif self.dataset == 'MNIST':
            MNIST(self.dataset_path, train=True, download=True)
            MNIST(self.dataset_path, train=False, download=True)
        elif self.dataset == 'CELEBA':
            ImageFolder(root=os.path.join(self.dataset_path, "celebA"))
        elif self.dataset == 'CARS':
            ImageFolder(root=os.path.join(self.dataset_path, "cars")) 

    def setup(self, stage=None):

        # Muovi il modello sul dispositivo solo qui, quando il dispositivo è corretto
        self.diffusion = DiffusionModel(device=self.device)
        self.model = Unet(self.model_config, self.device)

        if self.dataset == 'CIFAR10':
            train_dataset = CIFAR10(self.dataset_path, transform=self.transform, train=True)
            test_dataset = CIFAR10(self.dataset_path, transform=self.transform, train=False)
            self.full_dataset = ConcatDataset([train_dataset, test_dataset])
        elif self.dataset == 'MNIST':
            train_dataset = MNIST(self.dataset_path, transform=self.transform, train=True)
            test_dataset = MNIST(self.dataset_path, transform=self.transform, train=False)
            self.full_dataset = ConcatDataset([train_dataset, test_dataset])
        elif self.dataset == 'CELEBA':
            self.full_dataset = ImageFolder(root=os.path.join(self.dataset_path, "celebA"), transform=self.transform)
        elif self.dataset == 'CARS':
            self.full_dataset = ImageFolder(root=os.path.join(self.dataset_path, "cars"), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x0, _ = batch
        t = torch.randint(0, self.diffusion.n_steps, (x0.size(0),), dtype=torch.long).to(self.device)
        xt, noise = self.diffusion.q_xt_x0(x0.to(self.device), t)
        pred_noise = self.model(xt.float(), t)
        loss = F.mse_loss(noise.float(), pred_noise)
        self.losses.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        avg_train_loss = avg_train_loss = torch.mean(torch.tensor(self.losses[-len(self.full_dataset):]))
        self.epoch_losses.append(avg_train_loss.item())

        if avg_train_loss < self.best_loss:
            self.best_loss = avg_train_loss.item()
            self.best_epoch = self.current_epoch

        if (self.current_epoch) % self.save_image_epoch == 0:
            print(f"Generazione batch di {self.inference_batch_size} immagini")
            generate_images_grid(
                self.model, 
                self.current_epoch + 1,
                num_images=self.inference_batch_size,
                device=self.device,
                img_size=self.img_size,
                diffusion=self.diffusion,
            )

    def on_fit_end(self):

        with open('./result/losses.csv', 'w', newline='') as f_loss:
            writer = csv.writer(f_loss)
            writer.writerow(['Loss'])
            for loss in self.losses:
                writer.writerow([loss])

        with open('./result/epoch_losses.csv', 'w', newline='') as f_epoch_loss:
            writer = csv.writer(f_epoch_loss)
            writer.writerow(['Epoch', 'Loss'])
            for idx, loss in enumerate(self.epoch_losses, 1):
                writer.writerow([idx, loss])

        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Batch Loss')
        plt.title('Perdita per Batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('./result/batch_losses.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, label='Epoch Loss')
        plt.title('Perdita Media per Epoca')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('./result/epoch_losses.png')
        plt.close()


module = DiffusionLightningModule(
    model_config=model_config,
    dataset=dataset,
    dataset_path=dataset_path,
    img_size=img_size,
    batch_size=batch_size,
    lr=lr,
    patience=patience,
    save_image_epoch=save_image_epoch,
    inference_batch_size=inference_batch_size,
)

# Imposta il callback per il salvataggio del checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="checkpoint-{epoch:02d}-{train_loss_epoch:.4f}",
    monitor="train_loss_epoch",
    save_top_k=5,  # Salva solo il miglior modello
    mode="min",  # Minimizza la loss
    save_weights_only=False,  # Salva l'intero stato, non solo i pesi
    verbose=True,
)

# Impostazione dell'early stopping
early_stopping = EarlyStopping(
    monitor="train_loss_epoch",
    patience=patience,
    mode="min",
    verbose=True,
)

# Inizializza il trainer
trainer = Trainer(
    max_epochs=epochs,
    accelerator="gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    #strategy="ddp", # Per GPU o CPU
    devices=num_device,  # Numero di dispositivi (GPU, CPU o MPS)
    callbacks=[checkpoint_callback, early_stopping],
    logger=False,  
)

if os.path.exists(checkpoint_dir + "best_checkpoint.ckpt"):
    print("Carico l'ultimo checkpoint!")
    trainer.fit(module, ckpt_path=checkpoint_dir + "best_checkpoint.ckpt")
else:
    trainer.fit(module)
