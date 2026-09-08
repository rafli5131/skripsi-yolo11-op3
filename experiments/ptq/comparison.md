# Perbandingan Akhir OpenVINO (VAL)

> Metrik akurasi dan timing di bawah adalah hasil evaluasi diagnostik pada VAL. CPU utilization/RAM final harus diukur pada ROBOTIS OP3; timing host bukan benchmark deployment akhir.

Status QAT OpenVINO: **REJECTED / diagnostic metrics displayed — not valid for deployment**

| Metrik | FP32 OpenVINO | QAT INT8 | PTQ INT8 |
|---|---:|---:|---:|
| precision | 0.936223 | 0.934116 | 0.887335 |
| recall | 0.957564 | 0.950208 | 0.935881 |
| mAP50 | 0.972973 | 0.972145 | 0.961287 |
| mAP50-95 | 0.771344 | 0.767179 | 0.651042 |
| F1 | 0.946773 | 0.942093 | 0.910962 |
| Latency | 20.175 | 11.989 | 13.947 |
| FPS | 49.567 | 83.408 | 71.701 |
| CPU utilization | NOT MEASURED — requires controlled ROBOTIS OP3 benchmark | NOT MEASURED — requires controlled ROBOTIS OP3 benchmark | NOT MEASURED — requires controlled ROBOTIS OP3 benchmark |
| RAM | NOT MEASURED — requires controlled ROBOTIS OP3 benchmark | NOT MEASURED — requires controlled ROBOTIS OP3 benchmark | NOT MEASURED — requires controlled ROBOTIS OP3 benchmark |
| Ukuran model | 10,756,936 B | 3,407,480 B | 3,483,079 B |


## Delta QAT terhadap FP32

| Metrik | Delta absolut | Delta relatif |
|---|---:|---:|
| precision | -0.002107 | -0.225083% |
| recall | -0.007356 | -0.768156% |
| mAP50 | -0.000827 | -0.085048% |
| mAP50-95 | -0.004166 | -0.540062% |
| F1 | -0.004680 | -0.494301% |
