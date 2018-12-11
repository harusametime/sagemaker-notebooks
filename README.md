# sagemaker-notebooks

ここでは、公式の[Amazon SageMaker example](https://github.com/awslabs/amazon-sagemaker-examples)に加えて、いくつかの日本語のexampleを用意しています。ここのexampleに関して、コメントがある方、問題を発見された方は、このレポジトリの[issue](https://github.com/harusametime/sagemaker-notebooks/issues)より報告いただけるとありがたいです。

## 画像系

### chainer_mnist
Chainerでmnistの手書き数字の画像を分類します。これでSageMaker上での分散学習、ハイパーパラメータ最適化、ローカルモード実行、エンドポイントの起動・推論実行など一通り試すことができます。また、対象データセットが小さいため、ml.m4.xlargeといった小さめのインスタンスでも学習可能です。

### edge2food
食事画像の輪郭から食事画像をpix2pixで生成します。具体的な例として、food-101に含まれる1000枚のラーメン画像を使用して、輪郭情報を抜き出し、そこからラーメン画像を生成します。MXnet/Gluonを利用して実装しています。ネットワークが大きいためGPUでの学習を推奨します。エッジの抜き方が雑でまだ精度がでていません。

### transfer_learning
Amazon SageMakerのビルトインアルゴリズムImage classificationの転移学習を利用してCaltechデータセットを学習します。ビルトインアルゴリズムの利用にはp2, p3のインスタンスが必要です。

### tf_fine_tuning
SageMaker上でのTensorflowモデルのFine Tuningを行う例として、TF SlimのResnetモデルをS3にアップロードして、SageMakerでFine Tuningします。

### xgboost_mnist
XGBoostでmnistの手書き数字の画像を分類します。推論方法として、XGBoostのモデルをSageMaker外で読み込む方法と、SageMakerでエンドポイントを立てる方法を説明します。オープンソースのXGBoostとモデル互換性があるため、SageMaker外の環境（たとえば分析用のローカルマシンなど）で、モデルの簡単なテストや特徴量の重要度評価を行うことも可能です。

## 時系列データ

### random_cut_forest
異常検知用のビルトインアルゴリズムRandom Cut Forestを利用して、ニューヨークのタクシー乗車数の異常検知を行います。実際に異常を検知した時間帯において、特殊なイベントが開催されていたことを確認し、実際の異常検知でも利用可能であることを確認します。

## その他

### keras_mxnet
MXNetをバックエンドとしたKerasを利用して学習し、MXNetのモデルに変換してデプロイします。2018.12時点で、Kerasの複数インスタンスの学習およびKerasモデルを直接デプロイすることはできません。MNISTデータセットを対象としたMLPとCNNによる分類を行います。

### keras_tensorflow
TensorflowをバックエンドとしたKerasを利用して学習し、Tensorflow Serving用にモデルに変換してデプロイします。2018.12時点で、Kerasの複数インスタンスの学習およびKerasモデルを直接デプロイすることはできません。MNISTデータセットを対象としたMLPとCNNによる分類を行います。

### sklearn_classifier
re:Invent2018でScikit-Learnがデフォルトでサポートされるようになりました。公式githubをご覧ください。
https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/sklearn  
~~ChainerのSageMakerコンテナを利用してScikit-Learnの学習と推論を行います。~~

