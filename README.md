# visibility
- TODO
    - mesh.objの各頂点 v x y z で可視関数V(x,w)を求め可視化
    - pywavefrontを使うとpythonでobjファイルが読み込めるらしい
    - embreeのpython wrapperでレイトレーシングができるらしい
    - 各頂点から，レイを飛ばし可視関数の値を求める

- ``pywavefront``のインストール
    - pywavefront == 1.3.3 
  
```
pip install pywavefront
```
- ``pyembree``のインストール
  - python 3.6 ~ 3.9で対応？
  - pyembree == 0.1.12
```
pip install pyembree
```
上記のpythonのバージョン以外は
`embreex`パッケージを使用
```
pip install embreex
```

```
pip install cupy
```
HyperDreamerの ``sg_render.py``を基に改良を行っていく
