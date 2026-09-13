ICRA向け相互作用モード解析の論文構成・文献調査メモ

2026年9月10日。追加された解析コードと、集団入力の最大化・最小化に関する説明を反映。

**独立したICRA論文として構成できる。** 中心となる成果は、ペア実験から同定した身体由来の相互作用を、送信側と受信側の構造を持つ集団信号として表現し、自己組織化した位相関係を、各モジュールが受信する集団信号の振幅の増大・抑制、および極値との対応から理解することである。ペアモデルによる集団状態の再現に、この入力に基づく解釈を接続する。

今回の評価対象は、集団の位相関係、受信側で重み付けした集団信号、その極値、モード別寄与とする。遊泳速度とcost of transportは、制御に伴う実行周波数の変化がgaitの変化より大きく寄与している可能性があるとのユーザーの判断に従い、論文構成から除外する。

本メモの根拠は、MeetingWithAuke20260909(1).pdfの1–19ページ、DARS2026_Latebreaking (2).pdfの全2ページ、Optimal_Phase_Sensitivity_Functions_V2 (2)(3).pdfの全25ページ、およびPasted text(20260910-133943).txtである。これまでの議論については、この会話で確認できる説明を反映した。原データおよびSVD結果のCSVは添付されていないため、コードを実行して極値や予測精度を再計算したわけではない。入力振幅の最大化・最小化が確認されていることは、ユーザーが報告した既存の解析結果として扱う。

**コードが示している量を、論文の中心に置く。** 添付コードでは、観測した位相から

$$
x_k(t)=e^{i\phi_k(t)},\qquad
Z_\ell(t)=\sum_{k=1}^N v_{k\ell}x_k(t)
$$

を計算し、さらに受信側で重み付けした各成分とその和を

$$
G_{j\ell}(t)=\sigma_\ell u_{j\ell}Z_\ell(t),\qquad
G_j(t)=\sum_\ell G_{j\ell}(t)
$$

と定義している。W=UΣVᵀを使えば

$$
\boxed{G_j(t)=\sum_kW_{jk}e^{i\phi_k(t)}},\qquad
\boldsymbol G=W\boldsymbol x
$$

である。関数compute_agent_weighted_input_amplitude_for_fileは、1200–1203行でこの計算を行い、各モジュールの振幅|G_j(t)|を描いている。ここでの「sensitivityを掛ける」は、受信側の係数σ_ℓu_jℓを掛けることに対応する。

| コード内の式 | 数式上の意味 |
|---|---|
| `phase_phasors = exp(1i * phase_matrix)` | 各モジュールの複素位相信号 |
| `Z_complex = phase_phasors * V` | 各モードの送信側集団信号Z_ℓ |
| `G_complex = (Z_complex .* sigma.') * U.'` | 受信側で重み付けして、複素数のまま足したG_j |
| `G_abs = abs(G_complex)` | 各受信モジュールの合成信号の振幅 |

コードの「和」は、複素成分G_jℓを足してから絶対値を取るものである。各成分の振幅を足した量とは区別する。

$$
\left|\sum_\ell G_{j\ell}\right|
\ne\sum_\ell|G_{j\ell}|
\quad\text{（一般の場合）}.
$$

したがって、あるモードの振幅が大きくても、別モードとの位相関係によって受信側の合成振幅が小さくなることがある。この加算関係まで含めることが、「主要モードが大きくなる」という説明を具体的な入力の説明にする。

**ペア同定から、G_jが実機の感覚入力に現れるまでを示す。** 制御則を

$$
\dot\phi_j=\omega_j+\kappa z(\phi_j)s_j
$$

とし、ペア測定の感覚写像を

$$
s_{j\leftarrow k}(\phi_j,\phi_k)
\simeq\sum_{m,n=-M}^{M}C^{(j\leftarrow k)}_{mn}
e^{i(m\phi_j+n\phi_k)}
=s^{\mathrm{self}}_{j\leftarrow k}(\phi_j)
+s^{\mathrm{int}}_{j\leftarrow k}(\phi_j,\phi_k)
$$

とする。n=0が自己位相依存成分、n≠0が相互作用成分である。集団モデルは

$$
s_j(\boldsymbol\phi)
\simeq s_j^{\mathrm{self}}(\phi_j)
+\sum_{k\ne j}s_{j\leftarrow k}^{\mathrm{int}}(\phi_j,\phi_k)
$$

と構成する。自己項は、相手を変えた測定の整合性を確認して代表値を求め、一度だけ加える。

続いて

$$
s_{j\leftarrow k}^{\mathrm{int}}(\phi_j,\phi_k)
\simeq W_{jk}a(\phi_j)b(\phi_k),\qquad W_{jj}=0
$$

と分離する。全ペアで共通のa,bを用いるR=1の仮定と、Wの行列ランクは別である。R=1でもWは複数の特異モードを持つ。

15ページの波形に合わせて、送信関数を

$$
b(\phi)=B\cos(\phi-\delta)
$$

と定義すると、感覚入力の相互作用部分と、それによる位相速度の変化は

$$
s_j^{\mathrm{int}}(t)
\simeq B\,a(\phi_j(t))\operatorname{Re}\{e^{-i\delta}G_j(t)\},
$$

$$
\Delta\dot\phi_j^{\mathrm{int}}(t)
\simeq\kappa B\,z(\phi_j(t))a(\phi_j(t))
\operatorname{Re}\{e^{-i\delta}G_j(t)\}
$$

となる。G_jは、共通の受信位相プロファイルaを掛ける前の複素集団信号である。コードではa、z、κを掛けた瞬時感覚入力や瞬時位相速度変化そのものを描いているわけではない。これは入力の解析指標としての妥当性を否定するものではなく、各量の定義を区別するためである。

図のbの振幅は約√2、位相シフトδは26.2°と表示されている。δの符号は実装の定義に合わせる。共通δは|G_j|や|Z_ℓ|に影響しないので、今回のコードで振幅を評価するためにδを読み込む必要はない。実部を使って瞬時入力を再構成する際にはδが必要になる。

| 量 | ロボット系での役割 |
|---|---|
| v_kℓ | 各モジュールの位相が集団信号Z_ℓに寄与する係数 |
| σ_ℓu_jℓ | モードℓの集団信号をモジュールjが受ける係数 |
| G_jℓ | モジュールjに入るモードℓの複素信号 |
| G_j | 複数モードを加算した、モジュールjへの複素集団信号 |
| a(φ_j) | 身体・センサに由来する受信の位相依存性 |
| z(φ_j) | 制御器が指定する位相感受性 |

aとWは感覚相互作用の同定結果であり、zはソフトウェアで指定する制御変数である。GやZはオフライン解析の量で、実機がこれらを計算・通信する必要はない。

**入力の極値とモード解析を直接結ぶ式がある。** 受信する集団信号の二乗和を

$$
J_2(\boldsymbol\phi)=\sum_j|G_j(\boldsymbol\phi)|^2
$$

と定義する場合、Uの直交性から

$$
\boxed{
J_2=\|W\boldsymbol x\|_2^2
=\boldsymbol x^*W^\mathsf TW\boldsymbol x
=\sum_\ell\sigma_\ell^2|Z_\ell|^2
}
$$

が厳密に成り立つ。これは、受信側の集団信号の二乗和を、特異値で重み付けしたモード振幅の二乗和として表せるという関係である。この量の極値との対応が確認されていれば、入力の増大・抑制とモード別寄与を同じ数式の上で説明できる。

ただし、コードが直接描いている各jの|G_j|、その総和J_1=Σ_j|G_j|、二乗和J_2は別の評価量である。「最大化・最小化が確認されている量」を、論文内で一つずつ明記する。観測した指標を説明の都合でJ_2へ置き換えない。

各受信モジュールでは

$$
|G_j|^2=\sum_\ell|G_{j\ell}|^2
+2\sum_{\ell<m}\operatorname{Re}
\{G_{j\ell}\overline{G_{jm}}\}
$$

となり、モード間の相対位相も受信振幅を決める。全受信モジュールの二乗和を取ったときにはUの直交性によりこの交差項が消える。一方、さらにa(φ_j)、z(φ_j)を掛けた位相速度への寄与は、一般には直交しない。

また、全N本の右特異ベクトルを使えば

$$
\sum_\ell|Z_\ell|^2=N
$$

が恒等的に成り立つ。このため、重み付け前の|Z_ℓ|だけよりも、受信側のG_jやJ_2の変化を主結果に置くほうが、制御に関係する内容を明確にできる。これらの二乗量を、消費エネルギーなどの物理的エネルギーと同一視しない。

**「極値」の比較方法も、現在の成果に合わせて定義する。** 時系列上の増加・減少、収束後の局所極値、全位相配置に対する大域最大・最小、全初期条件からの収束保証は、それぞれ異なる主張である。既に確認した結果がどこまでを支持するかを、比較図と方法で示す。

固定Wに対し、各受信モジュール単独の振幅には次の厳密な上下限がある。

$$
U_j=\sum_k|W_{jk}|,\qquad
L_j=\max\left\{0,\,2\max_k|W_{jk}|-U_j\right\},
$$

$$
\min_{\boldsymbol\phi}|G_j|=L_j,\qquad
\max_{\boldsymbol\phi}|G_j|=U_j.
$$

これは複素平面上で固定長の各寄与を加算したときの幾何学から得られる。各jについて別々に位相配置を選んだ極値であり、同じ位相配置ですべてのjの極値を同時に達成できるとは限らない。

J_2には

$$
N\sigma_N^2\le J_2\le N\sigma_1^2
$$

というスペクトルによる評価がある。ただし、各|x_j|=1の制約があるため、この上下限が達成可能とは限らない。特異ベクトルの各成分の絶対値は一般に等しくなく、単一の特異ベクトルをそのままロボットの位相配置にはできない。

4モジュールなら、共通位相を固定した3次元の相対位相空間で、既存の極値探索結果と実機の収束状態を比較する図が作れる。候補解を数値探索で得た場合は、探索方法に応じて「得られた最良値」「局所極値」などと記述する。既に理論的な極値が分かっている場合はその値を使う。

**入力の極値との対応は、一般の勾配系の証明とは分ける。** A=WᵀWとおくと、J_2の位相勾配は

$$
\frac{\partial J_2}{\partial\phi_j}
=2\operatorname{Im}\{\overline{x_j}(A\boldsymbol x)_j\}.
$$

実装されているWinfree型フィードバックはWとz,aを通じて作用するため、一般にこの勾配と同じではない。G_jの極値に対応する状態が観測されることを示すために、勾配系である必要はない。一方、J_2の単調増加・減少や大域最適解への収束まで主張する場合には、別の条件・証明が必要となる。

特にW_jj=0の近似では、G_jはφ_jに直接依存しないので、∂|G_j|²/∂φ_j=0である。各モジュールが自分の位相を変えるだけで自身の|G_j|の勾配を直接上る、という説明は成立しない。自分の位相が他モジュールの入力と位相を変え、それが再び自分への入力を変える閉ループの作用として説明する。

**収束先の予測と、収束状態の入力に基づく解釈を組み合わせる。** SVDは、送信信号から受信信号への写像を分解する。一方、安定性や収束率は、自己位相依存項、a、z、自然周波数、κ、位相配置を含めた閉ループダイナミクスで決まる。特異値そのものは収束率ではない。

まず、ペア同定モデルから閉ループの集団位相関係を計算する。次に、予測された安定状態と観測された収束状態をG_jおよびモード成分で表し、既存の極値結果と比較する。この構成なら、予測能力と機構の解釈を混同せず、両方を示せる。

弱結合・近い自然周波数・一次平均化が妥当な場合は

$$
\Gamma_z(\psi)=\frac{1}{2\pi}\int_0^{2\pi}
z(\theta)a(\theta)b(\theta-\psi)\,d\theta
$$

を用い、平均化位相ϑ_jについて

$$
\dot\vartheta_j=\bar\omega_j+
\kappa\sum_{k\ne j}W_{jk}\Gamma_z(\vartheta_j-\vartheta_k),
\qquad
\bar\omega_j=\omega_j+\kappa\langle z s_j^{\mathrm{self}}\rangle
$$

と書ける。b(φ)=B cos(φ−δ)なら

$$
\Gamma_z(\psi)
=B\{C\cos(\psi+\delta)+S\sin(\psi+\delta)\},
\quad C=\langle za\cos\theta\rangle,\quad S=\langle za\sin\theta\rangle.
$$

これらは分離モデルからの導出で、新たな実験結果ではない。相対位相χ_j=ϑ_j−ϑ_1のN−1変数の方程式で平衡点とJacobianを調べる。固有値の実部がすべて負なら局所漸近安定である。初期位相からの予測には、複数初期値からの積分や吸引域の評価を組み合わせる。

κ=±5で平均化が十分正確かどうかは実際の位相速度変化で評価する。非平均化モデルでは、ロックしていても位相差が周期内で変動する場合がある。自己位相依存項が強い場合は、非平均化モデルの数値積分を主な予測に用い、必要に応じてPoincaré写像で周期軌道の安定性を調べる。査読中原稿18–20ページは、高ゲインと同定周波数からの変化によるモデル誤差を報告しており、適用範囲の議論に関係する。

**資料から確認できる成果を、原稿内の証拠として次のように整理する。**

| 根拠 | 確認できる内容 | 論文での役割 |
|---|---|---|
| スライド3–7ページ | 局所感覚フィードバックで4モジュールが協調し、κの符号によって位相関係が変化 | 解明したい集団挙動の提示 |
| 8–11ページ | 開ループのペア同定、自己項と相互作用項の分離、加法的な集団モデル | 身体相互作用の実験的なモデル化 |
| DARS 2ページ | ペア同定モデルが4モジュールの定常位相関係を再現したとの報告 | 集団予測の基礎。定量誤差と評価条件を追記する |
| 15–18ページ | 共通a,b、有向W、SVDと集団信号 | 送受信構造を持つ相互作用モードの定義 |
| 19ページ | κ=5では|Z_1|が大きく、κ=−5では|Z_2|と|Z_3|が大きい | 重み付け前のモード振幅の変化 |
| 追加コード | σ_ℓu_jℓを掛けて複素加算した各G_jの振幅を計算 | 受信側の集団信号の評価がすでに実装されている根拠 |
| ユーザーの追加説明 | sensitivityを掛けた信号またはその和の最大化・最小化が分かっている | 論文の主要な解析結果として採用し、該当する量・比較対象・図を明記する |

13ページの図は6ページと同じ実験位相図として表示されているため、これを低ランクモデルの予測精度を示す図に読み替えない。DARSの全ペアモデルの再現と、共通プロファイル近似・モード打切りによる再現は段階を分けて評価する。

17ページの画像内の特異値は約0.095、0.052、0.030、0.0012である。この丸め値から計算したWのFrobeniusノルム二乗の累積比は、1モードで約71.5%、2モードで約92.9%、3モードで約99.99%になる。これはWという行列の近似に関する値で、実機の位相予測精度ではない。論文では元のCSVから再計算する。

第4モードの|Z_4|が大きくても、特異値を掛けた寄与は小さくなり得る。一方、κ=−5では第3モードの|Z_3|が大きいため、特異値の順位だけで第3モードを捨てるのは適切でない。G_jの加算結果と、実機の位相配置に対する誤差で評価する。

**新規性は、既存手法の名前よりも、実験設定と分かったことに置く。** 「これまで、この種のシステムの収束先は予測できなかった」という一般的な主張は避ける。身体・制御器・環境の相互作用から生じる歩容の安定性を、実機のリターンマップから解析した研究がある。位相結合関数のデータ同定や、同期ダイナミクスのスペクトル解析も既存である。[Aoi et al., 2013](https://doi.org/10.1098/rsif.2012.0908)、[Kralemann et al., 2011](https://arxiv.org/abs/1102.3064)、[McGraw and Menzinger, 2008](https://doi.org/10.1103/PhysRevE.77.031102)

適切な問題設定は、「詳細な身体・流体モデルが利用できない局所感覚フィードバック系について、ペアの感覚同定から集団の位相協調を構成し、その収束状態を受信する集団信号の極値とモード別寄与から説明する」である。

| 優先的な先行研究 | 確認した成果 | 本研究との比較 |
|---|---|---|
| Owaki et al., 2013, “Simple robot suggests physical interlimb communication is essential for quadruped walking”, J. R. Soc. Interface. [著者所属機関の論文情報](https://kyushu-u.elsevierpure.com/en/publications/simple-robot-suggests-physical-interlimb-communication-is-essenti/) | 明示的に結合していない4発振器と局所力フィードバックで四足協調を生成 | 身体を介した協調の背景。本研究は実測相互作用から受信集団信号を復元・解釈する |
| Thandiackal et al., 2021, “Emergence of robust self-organized undulatory swimming based on local hydrodynamic force sensing”, Science Robotics. [EPFL掲載の抄録](https://graphsearch.epfl.ch/en/publication/24e7758d-d05d-4953-8955-6bdeaa9fc0cd) | 局所流体力フィードバックによる分節間協調と遊泳 | 流体を介した感覚協調の直接的な背景。本研究はペア写像・有向重み・集団入力の関係を同定する |
| Aoi et al., 2013, “A stability-based mechanism for hysteresis in the walk–trot transition in quadruped locomotion”, J. R. Soc. Interface. [論文](https://doi.org/10.1098/rsif.2012.0908) | 実機の脚間位相のリターンマップから歩容の安定性・双安定性を解析 | 安定性の解析自体を新規とはしない。本研究はペア感覚同定から集団モデルを構成する |
| Kralemann, Pikovsky and Rosenblum, 2011, “Reconstructing phase dynamics of oscillator networks”, Chaos. [著者原稿](https://arxiv.org/abs/1102.3064) | 多変量時系列から位相結合関数と有向結合を再構成 | Fourier同定自体は既存。本研究は局所感覚写像を開ループで同定し、zと分離して集団挙動を解析する |
| McGraw and Menzinger, 2008, “Laplacian spectra as a diagnostic tool for network structure and dynamics”, Physical Review E. [論文](https://doi.org/10.1103/PhysRevE.77.031102) | Laplacianの固有ベクトルを使って部分同期とモードごとの変化を解析 | モードによる同期の解釈は既存。本研究は実測した有向感覚重みのSVDから送信・受信を分け、各モジュールの入力を説明する |
| Zamboni, Owaki and Hayashibe, 2021, “Adaptive and Energy-Efficient Optimal Control in CPGs Through Tegotae-Based Feedback”, Frontiers in Robotics and AI. [本文](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.632804/full) | 制御状態と感覚反応から定義するTegotae関数と、その増大に対応する局所フィードバックを議論 | 「感覚に関係する量を大きくする局所制御」は先行例がある。本研究は、同定した物理相互作用から定まる集団入力と、その極値に対応するネットワーク位相状態を扱う |
| Laing, Bläsche and Means, 2021, “Dynamics of Structured Networks of Winfree Oscillators”, Frontiers in Systems Neuroscience. [本文](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2021.631377/full) | 構造を持つ有向Winfreeネットワークを秩序パラメータで解析 | Winfreeネットワークの集団解析は既存。本研究は有限個のロボットの実測感覚結合を扱う |
| Mastrogiuseppe and Ostojic, 2018, “Linking connectivity, dynamics and computations in low-rank recurrent neural networks”, Neuron. [著者原稿](https://arxiv.org/abs/1711.09672) | 低ランクの結合構造から集団活動を解析 | 低ランク結合と集団活動の関連づけの背景。ロボットの身体相互作用・局所感覚・実機検証の違いを示す |
| Lu, Maggioni and Tang, 2021, “Learning interaction kernels in heterogeneous systems of agents from multiple trajectories”, JMLR. [論文](https://jmlr.org/papers/v22/19-861.html) | 複数軌道からペア距離に依存する相互作用核を学習し、軌道を予測 | 相互作用を学習して集団を予測する一般的背景。今回の位相・局所感覚・ペア実験という設定を具体化する |
| Kulekcioglu et al., 2026, “Programmable Synchronization Graphs for Adaptive and Fault-Tolerant Modular Miniature Robots”. [プレプリントv2](https://arxiv.org/abs/2607.07281v2) | 符号付きグラフ結合でモジュラーロボットの位相関係を調整 | 最近の比較対象。同原稿はcoordinatorでプログラム可能なグラフ更新を実装する。本研究の身体由来の感覚相互作用とは制御対象が異なる |

入力振幅の極値という主張には、Tegotae研究との比較を加えることが重要である。あらかじめ設計したTegotae関数を使うことと、実機から同定したWによって定まるG_jを解析して集団の収束状態を説明することの違いを具体化する。今回の制御則を、そのままJ_2の勾配系と主張する必要はない。[Zamboni et al.のSection 2.1](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.632804/full)

Kralemannらは、元の物理系がペア結合であっても高次の位相縮約で多体の位相項が生じ得ることを説明する。ペア加算モデルは、そのような成分と集団化による力学条件の変化が十分小さい範囲で有効な近似として評価する。[著者原稿Section II](https://arxiv.org/pdf/1102.3064)

補助的には、単位複素数に関する二次形式の最適化・スペクトル緩和を扱うAmit Singerの“Angular Synchronization by Eigenvectors and Semidefinite Programming”も関連する。ただし、同研究の中心はノイズのある角度差測定からの角度推定で、今回の実機フィードバックの収束原理を直接証明する研究ではない。[著者原稿](https://arxiv.org/abs/0905.3174)

調べた範囲では、開ループのペア感覚同定、共通位相プロファイル、有向WのSVD、受信集団信号の極値、実機の集団位相状態を同じ設定で結びつけた研究は確認できなかった。これは網羅的な不在証明ではなく、論文では“the first”を使わずとも、比較対象との違いを具体的に主張できる。

**既存原稿との関係は、次のように整理する。**

| 成果 | 中心課題 | 今回の位置づけ |
|---|---|---|
| 既報RA-L | 受動関節の局所フィードバックで泳ぐモジュールを同期させる | ロボットと制御則の基礎として参照 |
| 査読中Inverse Design原稿 | 2体の周波数差に対するロック範囲を最大化するzの関数最適化 | 同定と制御器設計の背景。定理と2体のロック範囲改善を主要成果として区別する |
| DARS latebreaking | ペア感覚写像の合成で4体の位相関係を再現する | ユーザーの説明どおり、非アーカイバルの成果としてICRAの基礎に組み込む |
| 今回のICRA | 身体由来の相互作用を受信集団信号とモードで表し、集団位相状態と入力の極値との関係を説明する | ネットワーク構造、受信信号、極値、予測・実機検証を主要成果とする |

査読中原稿21ページには、ペア相互作用の加算によるネットワーク拡張が将来課題として記載されている。今回の共通プロファイル・SVD・受信集団信号の解析を同原稿が実証しているわけではない。z=±sinφを基本条件にすれば、2体の関数最適化とは異なる中心課題として構成できる。

ICRA2027の公式FAQは、正式・査読付きプロシーディングスのないワークショップ発表からの投稿を認め、DOI付きのアーカイバル発表は別扱いにしている。DARSについてはユーザーが示した非アーカイバルという前提で構成し、査読中原稿との問題設定・主要結果の違いを明記する。[ICRA2027公式募集・FAQ](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)

**追加作業は、既に得ている入力の極値に関する結果を提示することから始める。**

1. **既存の極値結果と収束状態を同じ図にする。** 各|G_j|、必要ならJ_1またはJ_2について、実験時系列・定常値・比較に使った極値を重ねる。フィードバック条件ごとに、どの量が最大化・最小化されているかを示す。コードは各|G_j|を既に計算しているので、この実装を未実施の追加作業として扱わない。
2. **モード別成分が合成振幅をどう作るかを示す。** G_jℓの振幅と相対位相、合成G_jを比較する。J_2を評価する場合は、σ_ℓ²|Z_ℓ|²による厳密な分解を利用する。これにより各モジュールの受信信号の増大・抑制を説明する。
3. **ペアモデルからの集団予測を定量化する。** ペアデータでa,b,Wと自己項を固定し、集団の閉ループ試行を評価用にする。定常位相差の円周距離、平均周波数、ロックの有無を比較する。集団試行に合わせて係数・ランクを調整した場合は、その試行を独立な予測検証とは呼ばない。
4. **近似段階とモードの必要性を比較する。** 自己項のみ、全ペアFourier、共通a,bと全W、1・2・3モードのモデルを比較する。特定モードを除いたときの入力の極値・収束状態も調べる。s全体は自己項が支配的なので、相互作用成分や位相速度への誤差も見る。
5. **必要な範囲で初期位相・制御条件に対する再現性を示す。** 既存の独立試行を優先する。追加するなら、同じL2ノルムのz(φ)=sin(φ+τ)で一つの未使用条件を予測して確かめる方法がある。複数ロボット種や移動性能の比較は、この論文の中心主張に必須ではない。

時系列の隣接サンプルをランダムに分けるだけの評価を避け、独立試行や独立した位相走査を単位として予測性能を評価する。既存データの反復数・計測条件を先に整理し、実験を増やすこと自体を目的にしない。

**コードから確認できた対応関係と、原稿化時の注意をまとめる。**

- 実機IDから同定側IDへの対応は、コードで9→8、8→10、11→7、12→9と明示され、UとVの両方に適用されている。したがって、スライド間のIDが異なることだけで誤りとは言えない。対応する幾何学的位置を原稿に示す。
- Gの計算は、列ベクトルの表記ではG=Wx、コードの時刻×モジュールの行列表記ではXWᵀとなり、転置の使い方は整合している。
- コード内のmode-sumとdirect-Wの比較は、同じU,Σ,Vから再構成したWを使う代数的チェックである。原実験とのモデル誤差やID対応の物理的正しさを検証するものではない。
- |G|の評価では共通δの省略は影響しない。瞬時の感覚入力や位相速度変化を再構成する場合には、δ、a、z、κを入れる。
- W_jj=0でも、個々のSVD成分や打切り行列の対角成分は一般に非零になる。モード除去・打切りで自己入力が混入するため、その扱いを定義する。対角を差し引く補正は可能だが、補正後の行列が同じランクであるとは限らない。
- 共通a,bの正規化と符号規約を固定する。特異値が近い場合は、単一モードの細かな差より対応する部分空間の再現性を評価する。
- スライド11ページの集団自己項に残るkの添字を整理し、代表のs_j^selfの推定方法を説明する。ストローク振幅aと受信関数a(φ)は別記号にする。

**8ページの原稿は、予測と受信集団信号の極値を中心に配分する。** ICRA2027の上限は、参考文献等を含めた8ページである。以下は構成案であり、会議指定の章構成ではない。[公式募集](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)

| 内容 | 配分の目安 | 示す内容 |
|---|---:|---|
| Introduction and Related Work | 1.0ページ | 具体的な予測課題、感覚に基づく協調とTegotae・同定・モード解析との比較 |
| Robot and Problem Formulation | 0.75ページ | 局所制御則、位相と感覚の定義、対象とする集団入力 |
| Identification of Physical Interactions | 1.0ページ | ペア計測、自己項の分離、共通a,b,W |
| Collective Signals and Interaction Modes | 1.5ページ | G_j、SVD、モード合成、極値・二乗和との関係 |
| Experimental and Numerical Evaluation | 1.75ページ | 実測対予測、入力の極値との対応、モード別寄与・除去 |
| Discussion and Conclusion | 0.5ページ | 有効範囲、モデル誤差、観測された極値と一般的最適化原理の区別 |
| References等 | 1.5ページ | 必要な文献と既報の位置づけ |

主要図は、(1)ロボットとペア計測、(2)自己項・相互作用項と共通プロファイル、(3)Wの送受信モード、(4)実機・モデルの集団位相関係、(5)受信集団信号と極値・モード成分の対応、の五つを中心にする。19ページの|Z_ℓ|だけで主結果を構成せず、追加コードの|G_j|と極値比較を主要図に置く。

**タイトルの第一候補は “From Pairwise Physical Interactions to Collective Phase Coordination in Modular Robots” である。** 入力の結果をより明示する候補は “Interpreting Self-Organized Phase Coordination through Collective Input Regulation in Modular Robots” である。未使用条件の予測を十分検証できれば、タイトルにPredictingを含めることもできる。

**Introductionの論理は、次の順序にするとよい。**

1. 身体・環境を介した局所感覚フィードバックによって、明示的な位相通信を使わずに集団のリズム協調が生じる。
2. 同じ形の制御則からどの位相関係が生じるかは、感覚に現れる物理相互作用によって決まり、その詳細な解析モデルを得ることが難しい。
3. ペアの開ループ感覚同定から集団モデルを構成し、共通受信・送信プロファイルと有向重みへ分離することで、身体相互作用を送受信構造を持つ集団信号として表せる。
4. 実機の収束状態と受信集団信号の極値との対応を示し、モード別寄与から入力の増大・抑制を説明する。ペアモデルによる集団状態の予測と合わせて、観測される協調を再現・解釈する方法を示す。

**英語abstractの暫定案は、追加説明を含めて以下のように書ける。** 最大・最小に関する文はユーザーが報告した既存結果に基づく。投稿版では、該当する評価量、比較した極値、独立試行の定量値を確定して記述する。

Local sensory feedback can generate collective rhythmic coordination through physical interactions among robotic modules. Predicting and interpreting the resulting phase patterns is difficult when the sensory interactions are not available as a tractable physical model. We investigate this problem in a swimming modular robot governed by local Winfree-type feedback. Open-loop pairwise experiments identify phase-dependent sensory interactions, which are separated into self-dependent and inter-module components and assembled into a collective phase model. Shared receiver and sender phase profiles and a directed weight matrix provide a structured representation of these interactions. A singular value decomposition then expresses the collective signal received by each module as a sum of weighted interaction modes. The pairwise model reproduces observed steady phase relations in a four-module robot. Analysis of the received collective signals associates the converged phase configurations with enhanced or suppressed input amplitudes and relates these states to the corresponding amplitude extrema. This representation connects collective phase coordination to the contributions and superposition of experimentally identified interaction modes. The results provide a basis for interpreting and predicting coordinated states from local sensory interactions without requiring explicit phase communication among modules.

**貢献は三点に絞る。** 第一に、ペア感覚同定から集団位相モデルを構成すること。第二に、身体の位相依存性と有向ネットワーク構造を分離し、各モジュールの受信集団信号を送受信モードで表現すること。第三に、自己組織化した位相関係を、受信信号の振幅の極値とモード別寄与に結びつけ、実機と同定モデルで検証することである。新しいSVDアルゴリズムや普遍的な大域収束則として主張する必要はない。

2026年9月10日時点の公式原稿締切は9月15日23:59 PST。今回の締切を目標とする場合は、既存の入力極値の結果を主要図として確定し、ペアモデル・分離モデルの予測誤差とモード除去の比較を優先する。公式案内上、動画には9月17–22日の提出期間もある。[日付と提出形式の公式情報](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
