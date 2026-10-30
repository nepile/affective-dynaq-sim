# The Big Picture

## Context

Halo, saya sedang membuat simulator untuk menguji agent Dyna-Q berbasis model emosi Valence-Arousal pada lingkungan gridworld (dyna maze). Saya ingin membuat simulator ini untuk pengujian saya dan kawan saya.

Pada paper Feldmaier dan Diepold menggunakan persamaan perhitungan emosi yang mereka buat dan mempengaruhi pembelajaran agent melalui modifikasi reward.

Untuk pengujian saya adalah tetap menggunakan persamaan perhitungan emosi tersebut, tetapi tidak memodifikasi reward yang diterima oleh agent. tetapi mempengaruhi parameter learning rate dan parameter strategi eksplorasi yang akan saya gunakan (epsilon-greedy, UCB, Boltzmann (Softmax) Temperature dan Posterior Sampling: Gaussian Thompson Sampling).

Untuk kawan saya adalah mengubah persamaan perhitungan emosi tersebut ke dalam ANN yang mempengaruhi reward yang diterima agent tanpa memodulasi parameter learning rate dan strategi eksplorasi yang digunakan.

Karena ini merupakan simulator penelitian, ada baiknya semua uji coba ada di settings dan hasilnya masuk ke reports. untuk saya sendiri memiliki beberapa eksperimen begitu juga teman saya. Oleh karena itu semua parameter atau konfigurasi masuk ke dalam file .cfg pada settings. file .cfg ini formatnya key-value.

Saya masih bingung untuk struktur project code ini sudah tepat atau bagaimana untuk menggabungkan ide tersebut dalam satu project yang sama tanpa mempengaruhi satu sama lain.

Setelah anda membaca markdown ini, tolong masuk ke file markdown berikutnya:
- tasks/task1.md