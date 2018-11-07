# Scikit-Learnを利用した分類
Amazon SageMakerでBring your own container (BYOC)なしに、scikit-learnの学習・推論を行うために以下を行います。
- コンテナ起動時にpipでscikit-learnをインストールする
  - pipでインストール可能なものについては、Docker fileの修正、ビルド、ECRへのpushなしに利用可能
  - ./souce_dirのなかにrequirements.txtをおくと、記載したライブラリ名がpipでインストールされます。scikit-learnをインストールする場合は以下のようにします。
  ```
  scikit-learn
  ```
- Chainerのコンテナイメージを流用する
  - chainerのコンテナイメージは推論時にnumpyを受け取ることができますので、通常のscikit-learnと同様に利用できます。
