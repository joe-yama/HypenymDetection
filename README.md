# GetHypernym

詳細：山根丈亮ら, 上位語・下位語の射影関係とそのクラスタの同時学習, 言語処理学会第22回年次大会発表論文集, pages 629-632, 2016

## データ
* vec300.txt  
Yahoo!知恵袋を教師データとし，学習した単語ベクトル (次元は300)

* model300.bin  
単語ベクトルと，学習した射影行列(バイナリファイル)


## コンパイル

```sh
make
```

## 環境

* g++4.8以上
* 2GB以上のRAM
* ライブラリ：Eigen <http://eigen.tuxfamily.org/>


## 使用法

### デモンストレーション

```
./GetHypernym -d MODEL_FILE DIMENSION  
```
MODEL_FILE : 単語ベクトルと学習した射影行列のファイル (model300.bin)  
DIMENSION : 単語ベクトルの次元 (300)

### 射影行列の学習

1. 教師データの読み込み(isa.binを出力)  

```
./GetHypernym -s vec300.txt training_data.txt isa.bin  DIMENSION
```

2. 射影行列の学習(MODEL_FILEを出力)  

```
./GetHypernym -g isa.bin NEGS THRESHOLD MODEL_FILE DIMENSION MAX_ITERATIONS  
```

NEGS : 負例の数 (推奨値：1)  
THRESHOLD : クラスタリングの際のしきい値 (推奨値：0.075)

### training_dataの体裁

```
'上位語\t下位語'  
```

の体裁で各行にひとつずつ上位下位語ペアを記述
