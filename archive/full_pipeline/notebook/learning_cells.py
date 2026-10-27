"""Executable teaching cells shared by the standalone notebooks."""


def learning_cells(kind, markdown, code):
    cells = [markdown("""## Laboratorium interaktif: jalankan sel satu per satu

Semua definisi model tersedia langsung di sel sebelumnya.
Sel di bagian ini benar-benar menjalankan model di kernel, tanpa unduhan dataset.
Setiap sel kode menampilkan hasil yang dapat diperiksa. Model sengaja kecil dan
menggunakan tokenizer karakter lokal, bukan tokenizer GPT-2 untuk training penuh.
Jalankan berurutan. Untuk mengulang eksperimen dari awal, restart kernel lalu Run all.
Variabel `model`, `xb`, `yb`, `logits`, `loss`, dan `optimizer` tetap tersedia.
"""), code(f'''
        import copy
        import torch
        import torch.nn.functional as F

        KIND = {kind!r}
        DEVICE = 'cpu'
        SEED = 1337
        torch.manual_seed(SEED)
        torch.set_printoptions(precision=3, sci_mode=False)
        TEXT = 'saya belajar model bahasa. model belajar menebak karakter berikutnya. '
        vocabulary = sorted(set(TEXT))
        stoi = {{char: index for index, char in enumerate(vocabulary)}}
        itos = {{index: char for char, index in stoi.items()}}
        cfg_demo = dict(n_head=2, n_embed=16, context_length=12,
                        vocab_size=len(vocabulary), N_BLOCKS=1)
        print('Eksperimen:', KIND, '| device:', DEVICE, '| PyTorch:', torch.__version__)
        print('Konfigurasi:', cfg_demo)
        print('Vocabulary:', stoi)
    '''), markdown("""### Data: input dan target bergeser satu posisi

`xb[b, t]` adalah karakter saat ini, sedangkan `yb[b, t]` karakter setelahnya.
Bentuk `[B,T]` berarti jumlah contoh dan panjang konteks. Target bukan salinan input.
"""), code('''
        encoded = torch.tensor([stoi[char] for char in TEXT], dtype=torch.long, device=DEVICE)
        samples = torch.stack([encoded[start:start + 13] for start in (0, 13)])
        xb, yb = samples[:, :-1], samples[:, 1:]
        assert xb.shape == yb.shape == (2, 12)
        assert torch.equal(xb[:, 1:], yb[:, :-1])
        print('xb:', xb, '\\nyb:', yb)
        print('Input :', ''.join(itos[i] for i in xb[0].tolist()))
        print('Target:', ''.join(itos[i] for i in yb[0].tolist()))
        print('Shape:', tuple(xb.shape), '| dtype:', xb.dtype)
    '''), markdown("""### Model dan embedding

Embedding token ditambah embedding posisi menghasilkan `[B,T,C]`.
Salinan dense disimpan untuk membandingkan output sebelum training pada input yang sama.
Konversi sparse mengganti MLP; attention tetap dense.
"""), code('''
        dense_model = Transformer(**cfg_demo).to(DEVICE)
        model = copy.deepcopy(dense_model)
        print('Parameter dense:', sum(p.numel() for p in dense_model.parameters()))
        print(dense_model)
    ''')]
    converters = {
        'pruned': 'sparsity=0.5', 'topk': 'active_fraction=0.25',
        'moe': 'num_experts=4, top_k=2', 'evolutionary': 'density=0.2',
    }
    if kind in converters:
        cells.append(code(f'''
            model = convert_model(model, {converters[kind]}).to(DEVICE)
            print('MLP:', model.attn_blocks[0].mlp)
            print('Parameter tersimpan:', sum(p.numel() for p in model.parameters()))
        '''))
    cells.extend([code('''
        with torch.no_grad():
            embedding = model._pre_attn_pass(xb)
            normalized = model.attn_blocks[0].ln1(embedding)
            head = model.attn_blocks[0].attn.heads[0]
            q, k, v = head.query(normalized), head.key(normalized), head.value(normalized)
            scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
            attention = scores.masked_fill(head.tril[:12, :12] == 0, float('-inf')).softmax(-1)
        assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)))
        assert torch.count_nonzero(attention.triu(1)) == 0
        print('Embedding:', tuple(embedding.shape), '| Q/K/V:', tuple(q.shape))
        print('Attention contoh pertama (baris=query, kolom=key):\\n', attention[0])
        print('Jumlah tiap baris:', attention[0].sum(-1))
    '''), markdown("""Matriks attention berbentuk segitiga: posisi awal tidak boleh melihat token masa depan.
Softmax membuat jumlah setiap baris menjadi 1. Sel berikut memeriksa MLP setelah
attention dan residual, tepat pada input yang dipakai blok asli.
"""), code('''
        with torch.no_grad():
            block = model.attn_blocks[0]
            residual = embedding + block.attn(block.ln1(embedding))
            mlp_input = block.ln2(residual)
            mlp_output = block.mlp(mlp_input)
        mlp = block.mlp
        print('MLP input/output:', tuple(mlp_input.shape), tuple(mlp_output.shape))
        print('Contoh output:', mlp_output[0, 0, :8])
    ''')])
    diagnostics = {
        'baseline': """
            with torch.no_grad():
                activations = mlp.relu(mlp.hidden(mlp_input))
            print('Hidden:', tuple(activations.shape))
            print('Fraksi nol setelah ReLU:', (activations == 0).float().mean().item())
            print('Nol dari ReLU bukan mask pruning; seluruh matriks tetap dihitung.')
        """,
        'pruned': """
            for name in ('hidden_mask', 'proj_mask'):
                mask = getattr(mlp, name)
                print(name, 'aktif:', int(mask.sum()), '/', mask.numel())
            print('Mask tetap; bobot kecil dinonaktifkan. Penyimpanan masih tensor dense.')
        """,
        'topk': """
            with torch.no_grad():
                activations = F.relu(mlp.hidden(mlp_input))
                values, indices = activations.topk(mlp.k, dim=-1)
            print('k:', mlp.k, '/', activations.shape[-1])
            print('Neuron terpilih token pertama:', indices[0, 0])
            print('Aktivasinya:', values[0, 0])
            print('Seluruh skor hidden tetap dihitung; nilai terpilih bisa nol setelah ReLU.')
        """,
        'moe': """
            with torch.no_grad():
                gate_logits = mlp.gate(mlp_input.reshape(-1, cfg_demo['n_embed']))
                gate_scores, chosen = gate_logits.topk(mlp.top_k, dim=-1)
                gate_weights = gate_scores.softmax(-1)
            print('Expert per token:', chosen)
            print('Bobot router token pertama:', gate_weights[0])
            print('Jumlah penugasan:', torch.bincount(chosen.flatten(), minlength=mlp.num_experts))
            print('Auxiliary loss:', mlp.aux_loss.item())
            print('Implementasi repo mengalikan kontribusi dengan top_k.')
        """,
        'evolutionary': """
            print('Koneksi aktif/total MLP:', active_connections(model))
            print('Mask awal acak; topologi akan berubah setelah langkah optimizer.')
        """,
    }
    cells.extend([code(diagnostics[kind]), markdown("""### Forward dan loss

`logits` mempunyai bentuk `[B,T,V]`: skor setiap token vocabulary pada tiap posisi.
Cross entropy membandingkan skor dengan target berikutnya. MoE menambahkan loss
penyeimbang router; bandingkan cross entropy terpisah dari total loss.
"""), code('''
        model.train()
        logits, ce_loss = model(xb, yb)
        aux_loss = (sum(b.mlp.aux_loss for b in model.attn_blocks)
                    if KIND == 'moe' else ce_loss.new_zeros(()))
        loss = ce_loss + 0.01 * aux_loss
        with torch.no_grad():
            dense_logits, dense_loss = dense_model(xb, yb)
        assert logits.shape == (2, 12, len(vocabulary))
        assert torch.isfinite(loss)
        print('Logits:', tuple(logits.shape))
        print('CE:', ce_loss.item(), '| aux:', aux_loss.item(), '| total:', loss.item())
        print('Dense CE sebelum training:', dense_loss.item())
        print('Selisih logits maksimum:', (logits.detach() - dense_logits).abs().max().item())
    '''), markdown("""### Backward: lihat gradien sebelum bobot berubah

Sel ini menghitung ulang forward agar aman dijalankan ulang setelah backward.
`zero_grad` menghapus akumulasi lama. `backward` menghitung turunan; optimizer belum
mengubah parameter. Aktifkan `DEBUG_ANOMALY` untuk melacak operasi backward bermasalah.
"""), code('''
        DEBUG_ANOMALY = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        with torch.autograd.set_detect_anomaly(DEBUG_ANOMALY):
            logits, ce_loss = model(xb, yb)
            aux_loss = (sum(b.mlp.aux_loss for b in model.attn_blocks)
                        if KIND == 'moe' else ce_loss.new_zeros(()))
            loss = ce_loss + 0.01 * aux_loss
            loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                print(name, '| grad norm:', round(parameter.grad.norm().item(), 6))
        before_update = model.token_embed.weight.detach().clone()
    '''), code('''
        optimizer.step()
        if KIND == 'evolutionary':
            enforce_model_masks(model)
        delta = (model.token_embed.weight.detach() - before_update).abs().max().item()
        print('Perubahan maksimum embedding:', delta)
        print('Jalankan ulang sel backward sebelum mengulang sel optimizer ini.')
    ''')])
    if kind == 'evolutionary':
        cells.append(code('''
            masks_before = [b.mlp.hidden_mask.clone() for b in model.attn_blocks]
            active_before = active_connections(model)
            grown = rewire_model(model, optimizer, fraction=0.3)
            active_after = active_connections(model)
            assert active_before == active_after
            print('Koneksi baru (hidden + projection):', grown)
            print('Aktif sebelum/sesudah:', active_before, active_after)
            print('Posisi mask hidden berubah:', sum(
                int((old != b.mlp.hidden_mask).sum())
                for old, b in zip(masks_before, model.attn_blocks)))
            print('Bobot koneksi baru nol; nilainya dipelajari pada langkah berikutnya.')
            print('Momen Adam maksimum pada posisi hidden yang berubah:',
                  optimizer.state[model.attn_blocks[0].mlp.hidden.weight]['exp_avg'][
                      masks_before[0] != model.attn_blocks[0].mlp.hidden_mask].abs().max().item())
        '''))
    cells.extend([markdown("""### Latihan pendek: menghafal satu batch

Loss per langkah ditampilkan. Penurunan pada batch yang sama hanya menunjukkan
optimisasi bekerja, belum membuktikan generalisasi. Gunakan data validasi terpisah
untuk eksperimen sesungguhnya. SET melakukan rewiring setiap lima langkah di demo ini.
"""), code('''
        history = []
        for step in range(1, 11):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, ce_loss = model(xb, yb)
            aux_loss = (sum(b.mlp.aux_loss for b in model.attn_blocks)
                        if KIND == 'moe' else ce_loss.new_zeros(()))
            loss = ce_loss + 0.01 * aux_loss
            assert torch.isfinite(loss), f'Loss tidak finite pada step {step}'
            loss.backward()
            optimizer.step()
            if KIND == 'evolutionary':
                enforce_model_masks(model)
                if step % 5 == 0:
                    rewire_model(model, optimizer, fraction=0.3)
            history.append(ce_loss.item())
            print(f'step={step:02d} CE={ce_loss.item():.4f} total={loss.item():.4f}')
        print('CE awal/akhir:', history[0], history[-1])
    '''), markdown("""### Generasi

Model mengambil distribusi token pada posisi terakhir, memilih satu token, lalu
mengulang dengan konteks yang diperpanjang. Model kecil ini belum menghasilkan
bahasa yang baik. Output berubah jika seed atau bobot berubah.
"""), code('''
        model.eval()
        torch.manual_seed(SEED)
        with torch.no_grad():
            generated = model.generate(xb[:1, :4], max_new_tokens=24)
        print('Token:', generated[0].tolist())
        print('Teks:', ''.join(itos[i] for i in generated[0].tolist()))
    '''), markdown("""### Debugging di notebook

- Setelah exception, jalankan `%debug` di sel baru. Gunakan `p xb.shape`, `p loss`,
  `up`, `down`, `n` (next), `s` (step), `c` (continue), dan `q` (keluar).
- Tambahkan `breakpoint()` tepat sebelum baris yang ingin diperiksa. Jangan
  meninggalkannya aktif ketika memilih Run all karena kernel menunggu input.
- Mengulang sel konfigurasi/model mereset bobot; mengulang loop melanjutkan training.

Dokumentasi: [IPython debugging dan magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html).
"""), code('''
        print('Kelas model dalam kernel:', type(model).__name__)
        print('Kelas MLP dalam kernel:', type(model.attn_blocks[0].mlp).__name__)
        print('Variabel tersedia: model, xb, yb, logits, loss, optimizer, history, attention')
        print('Untuk traceback interaktif setelah error: %debug')
    ''')])
    if kind == 'baseline':
        cells.extend(post_training_cells(markdown, code))
    return cells


def post_training_cells(markdown, code):
    return [markdown("""### Dari pretraining ke SFT, reward, DPO, PPO, dan GRPO

Contoh berikut menjalankan fungsi objective asli pada tensor kecil. Ini demonstrasi
matematika loss, bukan training RLHF lengkap. Training dengan dataset besar tersedia melalui skrip di repositori.

SFT hanya menghitung loss pada target jawaban. Mask mengikuti posisi target setelah
shift. Ubah posisi bernilai satu untuk melihat bagian mana yang dilatih.
"""), code('''
        demo_logits, _ = model(xb)
        response_mask = torch.zeros_like(xb)
        response_mask[:, 6:] = 1
        supervised_loss = sft_loss(demo_logits, xb, response_mask)
        print('Mask jawaban:', response_mask[0])
        print('Target aktif setelah shift:', int(response_mask[:, 1:].sum()))
        print('SFT loss:', supervised_loss.item())
    '''), markdown("""Reward model belajar mengurutkan dua jawaban. DPO langsung mengubah policy
berdasarkan selisih log-probabilitas chosen/rejected relatif terhadap reference.
Log-probabilitas di sini angka sintetis, bukan skor jawaban manusia.
"""), code('''
        chosen_rewards = torch.tensor([2.0, 0.5])
        rejected_rewards = torch.tensor([0.0, 1.0])
        preference_loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        preference_dpo, chosen_implicit, rejected_implicit = dpo_loss(
            torch.tensor([-2.0, -3.0]), torch.tensor([-4.0, -2.0]),
            torch.tensor([-3.0, -3.0]), torch.tensor([-3.0, -3.0]),
        )
        print('Reward loss:', preference_loss.item())
        print('Preference accuracy:', preference_accuracy(chosen_rewards, rejected_rewards).item())
        print('DPO loss:', preference_dpo.item())
        print('Implicit chosen/rejected rewards:', chosen_implicit, rejected_implicit)
    '''), markdown("""PPO membandingkan policy baru dengan policy pengumpul rollout. Rasio terlalu
besar dibatasi dalam objective. GRPO menghitung advantage dari reward beberapa
jawaban untuk prompt yang sama, lalu memakai objective token dengan penalti KL.
"""), code('''
        old_logp = torch.full((4, 3), -2.0)
        new_logp = old_logp + torch.tensor([0.3, -0.1, 0.0, 0.1])[:, None]
        action_mask = torch.ones_like(old_logp, dtype=torch.bool)
        rewards_demo = torch.tensor([0.0, 1.0, 0.5, 1.5])
        advantages = group_advantages(rewards_demo, group_size=4)
        ppo_demo, clip_fraction = ppo_policy_loss(
            new_logp, old_logp, advantages[:, None], action_mask,
        )
        grpo_demo, grpo_stats = grpo_loss(
            new_logp, old_logp, old_logp, advantages, action_mask,
        )
        print('Reward grup:', rewards_demo, '| advantage:', advantages)
        print('Rasio policy token pertama:', (new_logp - old_logp).exp()[:, 0])
        print('PPO loss:', ppo_demo.item(), '| clip fraction:', clip_fraction.item())
        print('GRPO loss:', grpo_demo.item(), '| stats:', grpo_stats)
    ''')]


CONCEPTS = {
'MLP': r'''MLP mengubah tiap vektor token secara terpisah: $h=\max(0,xW_1^\top+b_1)$, lalu $y=hW_2^\top+b_2$. $W_1\in\mathbb R^{4C\times C}$ dan $W_2\in\mathbb R^{C\times4C}$. ReLU memberi nonlinieritas; tanpa itu dua transformasi affine dapat digabung menjadi satu. Alternatifnya, GELU memberi transisi yang lebih halus.''',
'Head': r'''Satu head mencampur informasi antarposisi. $Q=XW_Q$, $K=XW_K$, $V=XW_V$, lalu $A=\operatorname{softmax}(QK^\top/\sqrt d+M)$ dan $O=AV$. Di sini $d$ lebar satu head; $M_{ij}=-\infty$ jika $j>i$ dan nol selainnya. Softmax bekerja per baris. Pembagi $\sqrt d$ mengendalikan skala dot product. Mask mencegah bocornya jawaban masa depan. [Paper attention](https://arxiv.org/abs/1706.03762).''',
'MultiHeadAttention': r'''Beberapa head membaca proyeksi yang berbeda. Hasilnya digabung pada sumbu fitur lalu diproyeksikan ke $C$: $O=\operatorname{concat}(O_1,\ldots,O_h)W_O$. Pilih $C$ habis dibagi jumlah head; konfigurasi demo memakai $16/2=8$.''',
'Block': r'''Blok memakai residual: $u=x+\operatorname{Attn}(\operatorname{LN}(x))$, $y=u+\operatorname{MLP}(\operatorname{LN}(u))$. LN adalah LayerNorm pada fitur satu token. Residual menyediakan jalur penjumlahan langsung untuk informasi dan gradien. Implementasi ini menormalisasi sebelum sublayer; arsitektur lain menaruh normalisasi sesudahnya.''',
'Transformer': r'''Token ID memilih baris embedding, bukan mengalikan nilai ID sebagai bilangan bermakna. Embedding posisi ditambahkan untuk membedakan urutan. Setelah blok, linear head menghasilkan logits $z\in\mathbb R^V$ dan $p_j=e^{z_j}/\sum_k e^{z_k}$. Loss next-token adalah $L=-\frac1{BT}\sum_{b,t}\log p_{b,t,y_{b,t}}$. `forward` mengembalikan pasangan `(logits, loss)`; loss `None` bila target tidak diberikan.

`forward_hidden` adalah jalur representasi akhir `[B,T,C]`. Metode lama `forward_embedding` mengeluarkan aktivasi internal lebar $4C$; jangan merangkainya untuk lebih dari satu blok karena blok berikut mengharapkan $C$. Demo tidak memakai metode lama itu. Gradient checkpointing berarti menghitung ulang aktivasi saat backward, tidak menyimpan berkas checkpoint. Generasi memakai sampling; argmax adalah alternatif deterministik.''',
'PrunedMLP': r'''Pruning mematikan bobot kecil dengan mask tetap: $W_{\mathrm{eff}}=W\odot M$. Elemen mask $M$ bernilai nol atau satu. `sparsity=0.5` mematikan separuh bobot setiap matriks, sementara bias tetap aktif. Tensor dan operasi linear masih dense, sehingga ini tidak membuktikan percepatan komputasi.''',
'TopKMLP': r'''Untuk setiap token, hitung seluruh aktivasi $h=\operatorname{ReLU}(W_1x+b_1)$, lalu pilih indeks $S=\operatorname{TopK}(h,k)$. Output $y=\sum_{j\in S}h_jW_{2,:,j}+b_2$. Pilihan neuron berubah sesuai input. Skor hidden tetap dihitung semua; top-k dapat mencakup nol jika ReLU mematikan banyak neuron.''',
'Expert': r'''Expert adalah MLP kecil yang mengolah token yang ditugaskan router. Lebarnya sebagian dari MLP asal, sehingga parameter dense dapat dibagi di antara expert.''',
'MoEMLP': r'''Mixture of Experts (MoE) memilih beberapa expert per token. Router menghasilkan skor $g(x)$; softmax hanya atas expert terpilih menghasilkan bobot $a_e$. Kode repo memakai $y=k\sum_{e\in S}a_e f_e(x)$, termasuk faktor $k$ yang perlu diperhatikan saat membandingkan implementasi lain. Ketika semua expert dipilih dengan bobot seragam, partisi bobot merekonstruksi MLP dense.

Loss tambahan $L_{aux}=E\sum_e I_eD_e$ memakai rata-rata probabilitas router $I_e$ dan fraksi penugasan $D_e$. Ini mendorong penggunaan lebih merata; keputusan top-k diskret tidak mendapat turunan langsung. [Paper MoE](https://arxiv.org/abs/1701.06538).''',
'EvolutionaryMLP': r'''Sparse Evolutionary Training (SET) mengganti koneksi saat latihan. Mask awal acak memiliki jumlah aktif tetap. Forward memakai $W\odot M$; sesudah optimizer, bobot nonaktif dinolkan kembali. Demo ini adalah varian sederhana: koneksi baru bernilai nol dan mulai belajar pada langkah berikutnya. [Paper SET](https://arxiv.org/abs/1707.04780).''',
'convert_model': 'Fungsi ini mengganti MLP setiap blok pada objek model yang diberikan, lalu mengembalikan objek yang sama. Karena perubahan terjadi langsung, demo menyalin model dense lebih dulu untuk menjaga pembanding.',
'evolutionary_mlps': 'Generator ini menghasilkan satu MLP SET per blok. `yield` menyerahkan objek satu per satu tanpa membuat daftar baru.',
'_clear_optimizer_state': 'Adam menyimpan momen gradien per bobot. Setelah koneksi diganti, momen posisi tersebut harus nol agar koneksi baru tidak mewarisi dorongan lama. Fungsi ini hanya mereset tensor state yang bentuknya sama dengan bobot; penghitung langkah skalar tetap utuh.',
'rewire_weight': r'''Pilih $m$ koneksi aktif dengan magnitudo terkecil untuk dihapus, lalu aktifkan $m$ posisi dari himpunan yang sebelumnya nonaktif. Karena jumlah yang dihapus dan ditambah sama, $\|M_{baru}\|_0=\|M_{lama}\|_0$. `fraction` menentukan bagian koneksi aktif yang diganti, dibatasi jumlah posisi nonaktif tersedia. Jika fraction positif dan ada ruang, minimal satu koneksi diganti.''',
'rewire_model': 'Terapkan rewiring ke matriks hidden dan projection pada setiap MLP. Hasil integer adalah jumlah koneksi baru, bukan jumlah seluruh posisi mask yang berubah (yang bernilai dua kalinya).',
'enforce_model_masks': 'Sesudah optimizer, fungsi ini memaksa bobot nonaktif menjadi nol. Mask tetap menentukan jalur yang menerima gradien pada forward berikutnya.',
'active_connections': 'Kembalikan pasangan jumlah bobot aktif dan jumlah seluruh elemen bobot MLP. Bias dan attention tidak dihitung dalam pasangan ini.',
'masked_mean': r'''Rata-rata bermask: $\bar x=\sum_i m_ix_i/\max(1,\sum_i m_i)$. Mask memilih posisi yang relevan. Pembagi minimal satu membuat mask kosong menghasilkan nol, bukan pembagian nol.''',
'sft_loss': r'''Supervised Fine-Tuning (SFT) melatih model pada contoh prompt dan jawaban. $L_{SFT}=-\sum_t m_{t+1}\log p(x_{t+1}\mid x_{\le t})/\max(1,\sum_t m_{t+1})$. Mask $m$ mengaktifkan target jawaban saja. Fungsi melakukan shift sendiri; masukkan token asli, bukan `yb` yang sudah bergeser. Demo mask enam posisi hanyalah contoh, bukan parser percakapan.''',
'bradley_terry_loss': r'''Reward model (RM) memberi skor skalar untuk satu jawaban. Untuk pasangan yang disukai dan ditolak, $L_{RM}=-\operatorname{mean}\log\sigma(r_+-r_-)$. Sigmoid $\sigma$ mengubah selisih skor menjadi probabilitas preferensi. Contoh memakai skor sintetis; tidak membangun reward model baru.''',
'preference_accuracy': 'Hitung fraksi pasangan dengan skor chosen lebih besar daripada rejected. Skor yang sama dianggap belum benar. Metrik ini tidak dipakai sebagai loss karena perbandingan diskret tidak memberi gradien yang berguna.',
'dpo_loss': r'''Direct Preference Optimization (DPO) menggunakan pasangan jawaban tanpa melatih reward model terpisah. $L=-\mathbb E\log\sigma(\beta[(\log\pi_+-\log\pi_-)-(\log\pi_{ref,+}-\log\pi_{ref,-})])$. $\pi$ adalah policy yang dilatih, reference dibekukan, dan $\beta$ mengatur skala perbandingan. Log-probabilitas dijumlahkan sepanjang jawaban. Reward implisit yang dikembalikan hanya diagnostik, dilepas dari grafik gradien dengan `detach`. [Paper DPO](https://arxiv.org/abs/2305.18290).''',
'ppo_policy_loss': r'''Proximal Policy Optimization (PPO) membatasi insentif perubahan policy melalui $L=-\mathbb E_m[\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A)]$, dengan $r=\exp(\log\pi_{baru}-\log\pi_{lama})$. Advantage $A$ mengukur hasil relatif terhadap baseline; tanda positif mendorong aksi, negatif menguranginya. Clipping bekerja pada objective, bukan menjamin semua probabilitas tetap di interval itu. Demo memakai advantage grup agar hitungannya kecil; PPO lengkap biasanya memakai value model dan estimasi advantage. [Paper PPO](https://arxiv.org/abs/1707.06347).''',
'group_advantages': r'''Group Relative Policy Optimization (GRPO) membandingkan beberapa jawaban untuk prompt yang sama. $A_i=(r_i-\bar r)/(s+\varepsilon)$. Kode memakai simpangan baku sampel ($G-1$ pada penyebut varians), sehingga ukuran grup harus lebih dari satu. Reward identik menghasilkan advantage nol. Urutan input harus mengelompokkan jawaban per prompt.''',
'k3_kl': r'''Untuk $d=\log\pi_{ref}-\log\pi_{baru}$, hitung $e^d-d-1\ge0$. Ini estimator penalti KL; kesetaraan ekspektasinya dengan KL memerlukan sampling dari policy baru dan dukungan distribusi yang sesuai. Angka sintetis demo hanya menunjukkan aljabar penalti.''',
'grpo_loss': r'''GRPO menggabungkan surrogate terpotong dengan penalti reference: $L=-\mathbb E_m[\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A)-\lambda K]$. $K$ berasal dari `k3_kl`, $\lambda$ adalah `kl_coef`, dan satu advantage jawaban disiarkan ke token-token jawaban itu. Kode merata-ratakan token aktif seluruh batch. RLHF berarti Reinforcement Learning from Human Feedback; SFT, DPO, PPO, dan GRPO merupakan pilihan tahap/metode, bukan urutan wajib semuanya. [DeepSeekMath, pengenalan GRPO](https://arxiv.org/abs/2402.03300).''',
}


def concept(symbol, kind):
    return f'### Membaca `{symbol}`\n\n' + CONCEPTS[symbol] + '''

<details><summary>Latihan membaca kode</summary>

Sebelum menjalankan, telusuri satu input sampai `return` (atau `yield`).
Apa yang tetap dan apa yang berubah pada bentuk tensor atau objeknya?
Cocokkan dengan tabel nama di bawah dan hasil demo setelah definisi model.
</details>
'''

# Curated descriptions are checked against every binding in the extracted code.
MEANINGS = {}

def _terms(names, description):
    for name in names.split():
        MEANINGS[name] = description

_terms('self', 'Objek instance kelas saat ini; atributnya bertahan antar pemanggilan metode.')
_terms('torch', 'Modul PyTorch: tensor, diferensiasi otomatis, dan optimizer.')
_terms('nn', 'Modul torch.nn untuk layer dan registrasi parameter model.')
_terms('F', 'Fungsi tensor neural network tanpa objek layer, misalnya linear dan cross entropy.')
_terms('Tensor', 'Tipe tensor PyTorch yang dipakai dalam anotasi fungsi.')
_terms('copy', 'Modul penyalin objek; deepcopy membuat parameter pembanding independen.')
_terms('math', 'Modul matematika Python untuk operasi skalar.')
_terms('checkpoint', 'Modul checkpoint aktivasi; menghitung ulang forward untuk menghemat memori.')
_terms('n_embed', 'Integer C: lebar representasi satu token.')
_terms('n_head', 'Integer jumlah head attention; C harus habis dibagi jumlah ini.')
_terms('context_length', 'Integer batas panjang konteks dan ukuran embedding posisi/mask causal.')
_terms('vocab_size V', 'Integer V: jumlah token berbeda yang dapat diprediksi.')
_terms('N_BLOCKS', 'Integer jumlah blok Transformer yang disusun berurutan.')
_terms('B', 'Integer jumlah contoh dalam batch.')
_terms('T', 'Integer panjang urutan pada pemanggilan ini.')
_terms('C', 'Integer dimensi terakhir tensor; pada attention adalah lebar input, pada Transformer.forward adalah V (logits).')
_terms('x', 'Tensor fitur input/intermediat [...,C]; dalam MLP dapat menjadi [...,4C] setelah hidden.')
_terms('idx tokens', 'Tensor integer [B,T] berisi ID token; bukan embedding.')
_terms('targets', 'Tensor integer target next-token; [B,T] sebelum flatten atau [B*T] untuk cross entropy.')
_terms('hidden', 'Aktivasi hidden tensor [...,H], atau atribut layer Linear(C,H); H=4C pada MLP dense.')
_terms('proj', 'Layer Linear(H,C) yang mengembalikan fitur hidden ke lebar embedding.')
_terms('relu', 'Layer ReLU yang mengganti nilai negatif dengan nol.')
_terms('key query value', 'Layer proyeksi fitur menjadi key/query/value satu head; value lokal di cleanup adalah tensor state Adam.')
_terms('tril', 'Buffer mask segitiga bawah [context_length,context_length], bukan parameter terlatih.')
_terms('head_size d', 'Integer lebar fitur satu head, C dibagi jumlah head.')
_terms('k q v', 'Tensor key/query/value [B,T,d] pada attention; k atribut TopKMLP adalah jumlah neuron terpilih.')
_terms('scale', 'Skalar 1/sqrt(d), pengali dot product attention.')
_terms('wei attention', 'Tensor bobot attention [B,T,T], nol di masa depan dan jumlah baris satu setelah softmax.')
_terms('out', 'Tensor hasil attention atau MLP, dengan dimensi akhir sesuai proyeksi yang dijalankan.')
_terms('heads', 'ModuleList berisi seluruh head; PyTorch mendaftarkan parameternya secara otomatis.')
_terms('h', 'Objek head saat iterasi list comprehension.')
_terms('attn', 'Objek MultiHeadAttention dalam satu blok.')
_terms('ln1 ln2 layer_norm', 'LayerNorm atas dimensi fitur; ln1 sebelum attention, ln2 sebelum MLP, layer_norm di akhir backbone.')
_terms('mlp', 'Objek MLP blok, dense atau varian sparse sesuai notebook.')
_terms('res residual', 'Tensor [B,T,C] jalur residual yang dijumlahkan dengan keluaran sublayer.')
_terms('token_embed', 'Embedding terlatih [V,C] yang memetakan ID token ke vektor.')
_terms('position_embed', 'Embedding posisi terlatih [context_length,C].')
_terms('attn_blocks', 'ModuleList berisi blok-blok Transformer berurutan.')
_terms('lm_head', 'Linear(C,V): proyeksi fitur terakhir menjadi logits vocabulary.')
_terms('pos_idxs', 'Buffer integer indeks posisi 0 sampai context_length-1.')
_terms('gradient_checkpointing', 'Boolean untuk menghitung ulang aktivasi saat backward; demo membiarkannya False.')
_terms('tok_embedding', 'Tensor embedding token [B,T,C].')
_terms('pos_embedding', 'Tensor embedding posisi [T,C], disiarkan ke seluruh batch saat penjumlahan.')
_terms('block b', 'Objek blok Transformer saat iterasi atau blok pertama dalam demo.')
_terms('logits', 'Skor sebelum softmax [B,T,V]; dalam DPO berupa selisih log-ratio [B], saat generasi diiris menjadi [B,V].')
_terms('flat_logits', 'Logits [B*T,V] untuk cross entropy.')
_terms('loss ce_loss dense_loss supervised_loss preference_loss preference_dpo ppo_demo grpo_demo', 'Tensor skalar objective; nama awalan menunjukkan CE, pembanding dense, SFT, reward, DPO, PPO, atau GRPO.')
_terms('max_new_tokens', 'Integer jumlah token tambahan yang di-sampling.')
_terms('idx_cond', 'Tensor ID konteks terakhir, panjangnya dibatasi context_length.')
_terms('probs', 'Tensor probabilitas next-token [B,V] setelah softmax.')
_terms('idx_next', 'Tensor integer [B,1] hasil sampling token berikutnya.')
_terms('_', 'Nilai atau indeks yang sengaja tidak digunakan.')
_terms('dense_mlp', 'Objek MLP asal; layer/bobotnya dipakai untuk inisialisasi varian sparse.')
_terms('sparsity', 'Float proporsi bobot yang dimatikan oleh pruning.')
_terms('density', 'Float proporsi koneksi aktif awal SET, antara nol dan satu.')
_terms('active_fraction', 'Float proporsi neuron hidden yang dipilih top-k per token.')
_terms('weight', 'Tensor parameter matriks bobot; bentuk [H,C] atau [C,H] pada MLP.')
_terms('mask hidden_mask proj_mask', 'Tensor mask: pada MLP memilih bobot, pada objective memilih token; bentuk mengikuti tensor yang disaring.')
_terms('count', 'Integer jumlah elemen yang dipilih pada pruning/rewire; pada masked_mean tensor skalar jumlah posisi aktif, minimal satu.')
_terms('width', 'Integer lebar hidden; lebar expert pada Expert, lebar MLP dense sebelum partisi pada MoEMLP.')
_terms('values', 'Tensor nilai top-k [...,k]; pada masked_mean tensor nilai yang hendak dirata-ratakan.')
_terms('indices', 'Tensor integer indeks neuron terpilih [...,k].')
_terms('columns', 'Tensor kolom projection terpilih [...,k,C] untuk penjumlahan kontribusi neuron.')
_terms('num_experts', 'Integer E: jumlah expert yang tersedia.')
_terms('top_k', 'Integer k: jumlah expert yang dipilih per token.')
_terms('expert_width', 'Integer lebar hidden satu expert, H/E.')
_terms('gate', 'Linear(C,E) tanpa bias: router memberi skor expert untuk setiap token.')
_terms('experts', 'ModuleList seluruh MLP expert.')
_terms('aux_loss', 'Tensor skalar penalti keseimbangan router; nol untuk demo non-MoE, None sebelum forward pertama MoEMLP.')
_terms('i index', 'Integer indeks saat enumerasi; i juga dipakai sebagai ID token saat decoding demo.')
_terms('expert', 'Objek MLP expert pada iterasi pembagian bobot atau routing.')
_terms('start end', 'Integer batas slice data atau batas bagian hidden milik expert; end eksklusif.')
_terms('flat', 'Tensor fitur [B*T,C], dimensi batch dan waktu digabung untuk routing.')
_terms('gate_logits', 'Tensor skor router [B*T,E] sebelum pemilihan expert.')
_terms('scores', 'Skor dot product [B,T,T] pada attention atau skor expert terpilih [B*T,k] pada MoE.')
_terms('chosen', 'Tensor integer [B*T,k] berisi indeks expert terpilih.')
_terms('weights gate_weights', 'Tensor bobot router [B*T,k], softmax atas expert terpilih.')
_terms('importance', 'Tensor [E]: probabilitas router rata-rata per expert, masih terhubung ke gradien.')
_terms('load', 'Tensor [E]: fraksi penugasan diskret per expert.')
_terms('output', 'Tensor akumulator kontribusi expert [B*T,C].')
_terms('expert_id', 'Integer indeks expert yang sedang diproses.')
_terms('token_idx slot_idx', 'Tensor integer indeks token dan slot routing yang memilih expert ini.')
_terms('contribution', 'Tensor kontribusi expert untuk token terpilih, sudah dikalikan bobot router dan top_k.')
_terms('model dense_model', 'Objek Transformer yang dilatih atau salinan dense pembanding sebelum training.')
_terms('optimizer', 'Objek AdamW dengan parameter, learning rate, dan momen gradien tersimpan di RAM.')
_terms('positions changed', 'Indeks posisi bobot yang diganti pada cleanup; changed di rewire_model adalah total jumlah koneksi baru.')
_terms('fraction', 'Float bagian koneksi aktif yang diganti per rewiring.')
_terms('flat_weight flat_mask', 'View satu dimensi dari bobot/mask; perubahan in-place memperbarui tensor asal.')
_terms('active dormant', 'Tensor integer posisi koneksi aktif/nonaktif; active di active_connections adalah jumlah aktif integer.')
_terms('weakest', 'Indeks lokal bobot bermagnitudo terkecil di antara koneksi aktif.')
_terms('removed grown', 'Indeks koneksi yang dihapus/ditumbuhkan di rewire_weight; grown pada demo adalah jumlah koneksi baru integer.')
_terms('total', 'Jumlah elemen bobot pada active_connections; jumlah nilai bermask tensor skalar pada masked_mean.')
_terms('KIND DEVICE SEED', 'Konfigurasi demo: string varian, string perangkat cpu, dan integer seed 1337 untuk generator acak.')
_terms('TEXT', 'String korpus kecil dalam RAM; seluruh kosakata berasal dari karakter di sini.')
_terms('vocabulary', 'List karakter unik yang diurutkan agar pemetaan ID konsisten.')
_terms('stoi itos', 'Dictionary karakter ke ID (stoi) dan ID ke karakter (itos).')
_terms('char', 'String satu karakter saat membangun vocabulary atau mengodekan teks.')
_terms('cfg_demo', 'Dictionary dimensi model demo: 2 head, C=16, konteks 12, satu blok, V sesuai vocabulary.')
_terms('encoded', 'Tensor integer satu dimensi berisi semua ID karakter korpus.')
_terms('samples', 'Tensor integer [2,13] sebelum dipisah menjadi input dan target.')
_terms('xb yb', 'Tensor integer [2,12]; xb input dan yb target yang bergeser satu karakter.')
_terms('p parameter', 'Tensor parameter model; .grad berisi turunan loss setelah backward.')
_terms('embedding normalized', 'Tensor [B,T,C] setelah penjumlahan embedding atau setelah LayerNorm sebelum attention.')
_terms('head', 'Objek head attention pertama yang dibuka untuk inspeksi.')
_terms('mlp_input mlp_output', 'Tensor [B,T,C] sebelum/sesudah MLP dalam jalur residual blok.')
_terms('activations', 'Tensor aktivasi hidden [B,T,4C] setelah ReLU.')
_terms('name', 'String nama parameter atau atribut mask dalam loop inspeksi.')
_terms('gate_scores', 'Tensor [B*T,k] skor router untuk expert terpilih.')
_terms('dense_logits demo_logits', 'Tensor [B,T,V] dari model pembanding dense atau input contoh SFT.')
_terms('DEBUG_ANOMALY', 'Boolean untuk pelacakan operasi yang menyebabkan backward bermasalah.')
_terms('before_update', 'Salinan tensor embedding sebelum optimizer.step, tanpa hubungan autograd.')
_terms('delta', 'Float maksimum perubahan absolut bobot embedding setelah satu langkah optimizer.')
_terms('masks_before', 'List salinan mask hidden sebelum rewiring, untuk membandingkan topologi.')
_terms('active_before active_after', 'Tuple integer (aktif,total) sebelum/sesudah rewiring; harus sama.')
_terms('old', 'Salinan mask lama saat membandingkan mask per blok.')
_terms('history', 'List float cross entropy setiap langkah; disimpan dalam RAM.')
_terms('step', 'Integer nomor langkah training dari 1 sampai 10.')
_terms('generated', 'Tensor integer [1,28]: empat token prompt ditambah 24 token hasil sampling.')
_terms('response_mask loss_mask resp_mask action_mask', 'Tensor mask token jawaban; hanya posisi aktif menyumbang objective.')
_terms('ce', 'Tensor cross entropy per target, kemudian dikalikan mask; [B,T-1] dalam SFT.')
_terms('chosen_rewards rejected_rewards', 'Tensor [B] skor reward jawaban disukai/ditolak; demo memakai dua pasangan.')
_terms('policy_chosen_logps policy_rejected_logps ref_chosen_logps ref_rejected_logps', 'Tensor [B] jumlah log-probabilitas token jawaban; policy dilatih, ref pembanding tetap; chosen/rejected menandai preferensi.')
_terms('beta', 'Float pengali log-ratio pada objective DPO, default 0.1.')
_terms('pi_logratios ref_logratios', 'Tensor [B] selisih log-probabilitas chosen dan rejected pada policy/reference.')
_terms('chosen_reward rejected_reward chosen_implicit rejected_implicit', 'Tensor [B] reward implisit beta*(log policy - log reference), detached untuk diagnostik.')
_terms('new_logp old_logp ref_logp', 'Tensor [B,L] log-probabilitas token aksi dari policy baru, pengumpul data lama, atau reference tetap.')
_terms('advantages', 'Tensor advantage [B] pada GRPO, atau [B,1]/[B,L] pada PPO; nilai positif berarti lebih baik dari baseline.')
_terms('clip', 'Float epsilon batas clipping rasio policy, default 0.2.')
_terms('ratio', 'Tensor [B,L] rasio probabilitas policy baru terhadap policy lama.')
_terms('surr1 surr2 surrogate', 'Tensor [B,L] surrogate tanpa clipping, dengan clipping, atau minimum keduanya.')
_terms('clipped', 'Tensor indikator rasio di luar interval clipping, dirata-ratakan sebagai diagnostik.')
_terms('group_size', 'Integer jumlah jawaban per prompt; harus lebih dari satu untuk std sampel.')
_terms('eps', 'Float konstanta kecil pada penyebut advantage untuk menghindari pembagian nol.')
_terms('rewards rewards_demo', 'Tensor satu dimensi reward, disusun per grup jawaban untuk prompt yang sama.')
_terms('r', 'Tensor reward [jumlah_prompt,group_size] setelah reshape; bukan rasio policy.')
_terms('mean std', 'Tensor [jumlah_prompt,1] rata-rata dan simpangan baku sampel reward per grup.')
_terms('adv', 'Tensor advantage yang dinormalisasi per grup atau diperluas menjadi [B,1] untuk broadcasting.')
_terms('diff', 'Tensor [B,L] selisih log-probabilitas reference minus policy baru.')
_terms('kl', 'Tensor [B,L] estimator penalti KL, satu nilai per token.')
_terms('kl_coef', 'Float lambda pengali penalti KL terhadap reference, default 0.04.')
_terms('per_token', 'Tensor [B,L] surrogate dikurangi penalti KL sebelum masked mean dan tanda negatif.')
_terms('stats grpo_stats', 'Dictionary float statistik KL rata-rata dan fraksi rasio terpotong; bukan state optimizer.')
_terms('clip_fraction', 'Tensor skalar fraksi token aktif dengan rasio keluar dari interval clipping.')

FUNCTIONS = {
'__init__': 'Membangun layer dan buffer, memeriksa konfigurasi; tidak mengembalikan tensor.',
'forward': 'Menghitung keluaran layer/model dari input. Dipanggil otomatis oleh objek(input); bentuk dan rumus dijelaskan di atas.',
'forward_embedding': 'Mengembalikan aktivasi hidden internal; pada Block/Transformer juga residual. Ini jalur inspeksi, bukan forward logits.',
'project_embedding': 'Memproyeksikan aktivasi hidden lebar 4C ke C melalui layer projection.',
'_pre_attn_pass': 'Mengembalikan jumlah embedding token dan posisi [B,T,C].',
'forward_hidden': 'Menjalankan seluruh blok dan normalisasi akhir; hasil [B,T,C].',
'generate': 'Mengambil sampel satu token berulang kali; hasil ID prompt ditambah max_new_tokens.',
'_random_mask': 'Membuat mask boolean acak dengan jumlah aktif sesuai density dan bentuk bobot.',
'_magnitude_mask': 'Membuat mask dengan mematikan bobot terkecil menurut magnitudo.',
'enforce_masks': 'Mengalikan bobot dengan mask secara in-place tanpa autograd; hasil None.',
'masked_layers': 'Generator pasangan (weight,mask) untuk hidden dan projection.',
}


def explain_code(source):
    import ast
    tree = ast.parse(source)
    entries = set()
    functions = []

    def walk(node, scope='kernel'):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            full = scope + '.' + node.name if scope != 'kernel' else node.name
            functions.append((full, node.name))
            for child in ast.iter_child_nodes(node):
                walk(child, full)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            entries.add((scope, node.id))
        elif isinstance(node, ast.arg):
            entries.add((scope, node.arg))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            entries.add((scope, node.attr))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                entries.add((scope, alias.asname or alias.name.split('.')[0]))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'register_buffer':
            if node.args and isinstance(node.args[0], ast.Constant):
                entries.add((scope, node.args[0].value))
        for child in ast.iter_child_nodes(node):
            walk(child, scope)
    walk(tree)
    missing = {name for _, name in entries if name not in MEANINGS}
    missing.update(name for _, name in functions if name not in FUNCTIONS and name not in CONCEPTS)
    if missing:
        raise ValueError('Penjelasan belum tersedia: ' + ', '.join(sorted(missing)))
    rows = ['<details><summary>AntiVibe: arti setiap nama dan fungsi dalam sel ini</summary>', '',
            'Scope `kernel` bertahan antar sel; scope metode/fungsi lokal dibuat ulang saat dipanggil. Atribut `self` melekat pada objek. Ukuran simbolis mengikuti pengantar.', '',
            '| Scope | Nama | Arti, tipe, dan bentuk |', '| :--- | :--- | :--- |']
    for scope, name in sorted(entries):
        rows.append(f'| `{scope}` | `{name}` | {MEANINGS[name]} |')
    if functions:
        rows += ['', '| Fungsi atau kelas | Peran |', '| :--- | :--- |']
        for full, name in functions:
            explanation = FUNCTIONS.get(name, CONCEPTS.get(name, '').split('. ')[0] + '.').split('\n')[0]
            rows.append(f'| `{full}` | {explanation} |')
    return '\n'.join(rows) + '\n\n</details>\n'

_terms('attn_weights', 'Tensor [B,T,T] skor attention yang kemudian dimask dan dinormalisasi softmax.')
_terms('scale_factor', 'Float faktor 1/sqrt(d) untuk mengendalikan skala dot product.')
FUNCTIONS['_mask'] = FUNCTIONS['_magnitude_mask']

PRACTICE = {
'### Data:': r'''Contoh: teks `saya` memberi input `say` dan target `aya`. Tokenizer karakter memudahkan inspeksi; tokenizer subword dapat merepresentasikan potongan kata dan biasanya menghasilkan urutan yang lebih pendek.

<details><summary>Latihan: mengapa panjang input 12 memerlukan 13 karakter?</summary>

Target terakhir adalah karakter setelah input terakhir. Dua slice sepanjang 12 harus diambil dari urutan sepanjang 13; target sudah bergeser, jadi forward model tidak boleh menggesernya lagi.
</details>''',
'### Backward:': r'''Autograd menerapkan aturan rantai pada grafik operasi: jika $L=L(z)$ dan $z=f(\theta)$, maka $\nabla_\theta L=J_f(\theta)^\top\nabla_zL$. Parameter $\theta$ adalah bobot terlatih. `backward()` mengisi `.grad`; `optimizer.step()` baru mengubah bobot. AdamW memakai rata-rata bergerak gradien dan kuadrat gradien, serta weight decay terpisah. Ini lebih adaptif daripada langkah SGD $\theta\leftarrow\theta-\eta\nabla_\theta L$; $\eta$ adalah learning rate. [Autograd PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).

<details><summary>Latihan: apakah grad norm nol berarti kodenya rusak?</summary>

Belum tentu. Mask pruning, ReLU yang mati, atau expert yang tidak terpilih dapat membuat sebagian gradien nol. Yang diperiksa adalah gradien finite, loss terhitung, dan ada bobot yang berubah. `NaN` dan tak hingga berbeda dari nol.
</details>''',
'### Latihan pendek:': r'''<details><summary>Latihan: apakah loss akhir lebih kecil membuktikan model memahami bahasa?</summary>

Tidak. Sepuluh langkah memakai batch yang sama, sehingga model dapat menghafal. Ukur loss pada teks terpisah untuk menilai generalisasi. Tidak ada jaminan loss turun pada setiap langkah, terutama ketika SET mengganti topologi.
</details>''',
'### Generasi': r'''<details><summary>Latihan: mengapa bentuk keluaran menjadi [1,28]?</summary>

Ada satu prompt dengan empat karakter. `generate` menambahkan 24 ID sehingga panjang akhirnya 28. Setiap prediksi hanya melihat paling banyak 12 ID terakhir; hasil yang dikembalikan tetap memuat seluruh urutan.
</details>''',
}
