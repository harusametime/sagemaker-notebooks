# Scikit-Learnを利用した分類
Amazon SageMakerでBring your own container (BYOC)なしに、以下の手順でscikit-learnの学習・推論を行います。
- コンテナ起動時にpipでscikit-learnをインストールする
  - pipでインストール可能なものについては、Dockerfileの修正、ビルド、ECRへのpushなしに利用可能
  - ./souce_dirのなかにrequirements.txtをおくと、記載したライブラリ名がpipでインストールされます。scikit-learnをインストールする場合は以下のようにします。
  ```
  scikit-learn
  ```
- Chainerのコンテナイメージを流用する
  - chainerのコンテナイメージでは、推論時の入力としてnumpyを受け取る関数がデフォルトで定義されており、通常のscikit-learnと同様に利用できます。もちろん、jpegなどのファイルを受け取るように、content-typeごとに関数を定義することも可能です。
