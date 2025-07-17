# ShapeGAN
A GAN which tries to recreate geometric shapes, its output is conditioned on the same parameter which generated the shapes in the training data.
Since there is no "noise" in the dataset, I took out the variational part of the GAN.


Example Training data generated from shape_clean_small.py

<img width="1472" height="1477" alt="sample_visualization" src="https://github.com/user-attachments/assets/d041e33b-6640-4d6d-b107-e7347533d6ac" />

In Progress Training:

Epoch 10...

<img width="1416" height="1447" alt="epoch_010" src="https://github.com/user-attachments/assets/407d6346-703b-42ba-9198-63fd3672c2b3" />


Epoch 295:

<img width="1416" height="1447" alt="epoch_295" src="https://github.com/user-attachments/assets/6a78df2c-5830-4f49-b400-14df8342a32f" />

Not quite there .. yet?



Once Trained, load the best_model.pth with  Generator_UI.py

**note** screenshot is of an earlier version with the Noise latent, which is now removed

<img width="796" height="629" alt="Screenshot 2025-07-15 202143" src="https://github.com/user-attachments/assets/078d7474-3a6d-46dc-ba2b-bae996879c02" />
