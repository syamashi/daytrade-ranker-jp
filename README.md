# デイトレ候補ランキング（試作版）

日本株30銘柄を対象に、5分足から候補を点数化し、GitHub Pagesへ表示します。

評価要素：直近1時間の騰落、出来高増加、VWAP乖離、売買代金。極端な前日比には過熱減点を加えます。

## GitHubで公開する

1. このフォルダを新規GitHubリポジトリへpush
2. リポジトリの **Settings → Pages** を開く
3. **Deploy from a branch**、`main`、`/docs` を選択
4. **Actions → Update ranking → Run workflow** を実行

平日の9:05～15:05（日本時間）、1時間ごとに自動更新されます。GitHub Actionsの混雑時は遅れることがあります。

## PCで試す

```powershell
py ranker.py
Start-Process docs/index.html
```

## 注意

Yahoo Financeの非公式エンドポイントを使う試作品です。取引所公式データではなく、遅延・欠損・停止の可能性があります。自動注文は実装していません。
