# Epsilon-Greedy Exploration Strategy

## Context

Sekarang fokus pada pembuatan strategi eksplorasi epsilon-greedy. tempatkan file code nya di src/policies/epsilon_greedy.py. Lalu nanti untuk setup parameter atau konfigurasi strategi eksplorasi epsilon-greedy harus melalui file konfigurasi (.cfg). 

Nantinya untuk masing-masing strategi eksplorasi (epsilon-greedy, UCB, Boltzmann/Softmax Temperature dan Gaussian Thompson Sampling) memiliki 3 mode yaitu mode static (yang berarti parameter akan tetap sama dari awal hingga episode berakhir pada masing-masing seed lingkungan). lalu ada mode exponential_decay (yang berarti parameter akan di set tinggi dan akan meluruh/menyusut secara eksponensial sampai episode berakhir dalam 1 lingkungan seed. semisal 1 lingkungan seed ini sudah selesai dan berpindah ke lingkungan seed berikutnya, parameter eksplorasinya kembali ke parameter init (nilai parameter yang tinggi) atau istilah kata lain yaitu di reset). Dan mode terakhir adalah adaptive_va yang dimana parameter eksplorasinya dimodulasi atau seperti stabilizer untuk menaik dan turunkan parameter eksplorasinya tergantung emosi Valence-Arousal, tapi untuk sekarang mode adaptive_va belakangan saja (belum perlu untuk sekarang).

Begitu juga dengan learning_rate yang memiliki 3 mode yaitu static, exponential_decay dan adaptive_va. untuk penjelasan ke-3 mode tersebut sama dengan strategi eksplorasi. untuk mode adaptive_va juga belum perlu untuk sekarang.

Pastikan untuk penggunaan mode adaptive_va yang hanya bisa digunakan pada lingkungan classic_va (gridworld_va_classic.py). Untuk lingkungan lainnya seperti standard dan ann, hanya boleh menggunakan 2 mode yaitu static atau exponential_decay.