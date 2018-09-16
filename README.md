# sagemaker-notebooks
### chainer_mnist
ChainerでMnistの手書き画像を分類します。これでSageMaker上での分散学習、ハイパーパラメータ最適化、
ローカルモード実行、エンドポイントの起動・推論実行など一通り試すことができます。また、対象データセットが小さい
ため、ml.m4.xlargeといった小さめのインスタンスでも学習可能です。

### edges2food
食事画像の輪郭から食事画像をpix2pixで生成します。具体的な例として、food-101に含まれる1001枚のラーメン画像を使用して、輪郭情報を抜き出し、そこからラーメン画像を生成します。MXnet/Gluonを利用して実装しています。

### transfer_learning
Amazon SageMakerのビルトインアルゴリズムImage classificationの転移学習を利用してCaltechデータセットを学習します。ビルトインアルゴリズムの利用にはp2, p3のインスタンスが必要です。


