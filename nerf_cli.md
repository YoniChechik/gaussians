ns-download-data nerfstudio --capture-name=poster

ns-train splatfacto --data /tmp/main_res/20250214_203539 
<!-- ns-train splatfacto --load-config /root/gaussians/outputs/20250214_203539/splatfacto/2025-02-14_203852/config.yml --load-checkpoint /root/gaussians/outputs/20250214_203539/splatfacto/2025-02-14_203852/nerfstudio_models/step-000004000.ckpt
ns-train splatfacto --data /tmp/main_res/20250214_203539 --load-dir /root/gaussians/outputs/20250214_203539/splatfacto/2025-02-14_203852/nerfstudio_models -->
ns-viewer --load-config outputs/20250214_203539/splatfacto/2025-02-14_211447/config.yml 
