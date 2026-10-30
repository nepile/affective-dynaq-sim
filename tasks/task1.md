# Create Environment Gridworld for Dyna-Q agent

## Context

Buatkan saya 2 environment:
1. environment biasa tanpa monitoring f(col), f(right), f(dir) dan f(goal) untuk dikirim ke affect untuk dihitung Valence-Arousal. ini saya butuhkan untuk pengujian dyna-q standar yang menggunakan static parameter dan eksponensial decaying parameter. (menurut anda lingkungan ini cocok untuk lingkungan teman saya yang menggunakan ANN?)
2. environment yang bisa memonitoring agent untuk menghitung nilai emosi Valence-Arousalnya yang nantinya akan berguna sebagai sinyal menaikkan atau menurunkan learning rate dan parameter eksplorasi tergantung yang strategi pemilihan aksi yang digunakan.

Untuk semua environment yang ada kami menggunakan generator dan distribusi obstacle yang sama yaitu Randomized Kruskal's Algorithm dan Cellular Automata yang nantinya titik start dan goal serta verifikasi ketersediaan jalur akan menggunakan Maximum Geodesic Distance. Untuk pengujiannya juga bukan hanya pada satu fixed seed, namun multi-fixed seed independen. Sebagai contoh jika di file konfigurasi saya input seed=[1, 3, 5, 6], berarti untuk masing-masing lingkungan mempunyai 6 lingkungan dengan obstacle dan titik start dan goal yang berbeda. Sebagai contoh lagi Untuk 1 lingkungan saya berikan agent beberapa episode, setelah lingkungan dengan seed 1 selesai dan telah menghasilkan report, reset q-table dan agentnya kembali seperti semula untuk menjalankan lingkungan dengan seed 3 (itu maksudnya multi-fixed seed independ).

Pada folder ui/gui.py itu untuk menampilkan mode gui, jika ui/headless.py menampilkan dibalik layar dan dijalankan di terminal.

Saya ingin baik dijalankan dalam mode gui atau headless, ketika program running tolong simpan gambar dari lingkungan yang sudah di beri obstacle dan titik start serta goalnya pada masing-masing seed. Simpan ke dalam reports dan berikan nama sub folder sesuai obstacle generatornya lalu nama file gambarnya sesuai lingkungan dengan seed berapa (visualisasi lingkungan untuk paper saya).

Pastikan dokumentasi dalam berbahasa inggris. Perlu anda ketahui saya sudah setup sphinx untuk autodoc generate dokumentasi dari src. Jadi tolong ikuti standar dokumentasi sphinx untuk menghasilkan dokumentasi yang bisa dibaca, dimengerti oleh orang awam, kawan saya dan saya sendiri.

Pastikan yang dijalankan hanya main.py yang dimana membaca konfigurasi untuk nilai yang dibutuhkan script lain diambil dari file .cfg.
