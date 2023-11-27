import numpy as np
import pandas as pd

def load_data(index_path, train_steps):
    index = pd.read_csv(index_path)
    index = ~index.isna()  # 更改为 ~index.isna() 以得到一个布尔数组，而不是np.isnan()

    dbz, y, zdr, kdp = [], [], [], []

    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            if index.iloc[i, j]:
                try:
                    dbz_temp, zdr_temp, kdp_temp = [], [], []
                    for k in range(train_steps):
                        dbz_path = f'/root/autodl-tmp/competition/data/NJU_CPOL_update2308/dBZ/Processed_3.0km/data_dir_{str(i).rjust(3, "0")}/frame_{str(j+k).rjust(3, "0")}.npy'
                        zdr_path = f'/root/autodl-tmp/competition/data/NJU_CPOL_update2308/ZDR/Processed_3.0km/data_dir_{str(i).rjust(3, "0")}/frame_{str(j+k).rjust(3, "0")}.npy'
                        kdp_path = f'/root/autodl-tmp/competition/data/NJU_CPOL_update2308/KDP/Processed_3.0km/data_dir_{str(i).rjust(3, "0")}/frame_{str(j+k).rjust(3, "0")}.npy'

                        dbz_temp.append(np.load(dbz_path))
                        zdr_temp.append(np.load(zdr_path))
                        kdp_temp.append(np.load(kdp_path))

                    y_path = f'/root/autodl-tmp/competition/data/NJU_CPOL_update2308/dBZ/Processed_3.0km/data_dir_{str(i).rjust(3, "0")}/frame_{str(j+10).rjust(3, "0")}.npy'
                    y.append(np.load(y_path))
                    dbz.append(dbz_temp)
                    zdr.append(zdr_temp)
                    kdp.append(kdp_temp)

                except Exception as e:
                    print(f'Error at {i}, {j}, {k}: {e}')

    return np.array(y), np.array(dbz), np.array(zdr), np.array(kdp)

# 调用函数
index_path = '/root/autodl-tmp/competition/util/sample.csv'  # 请替换为您的index_path
train_steps = 10  # 请替换为您的train_steps
y, dbz, zdr, kdp = load_data(index_path, train_steps)
print(y.shape, dbz.shape, zdr.shape, kdp.shape)
