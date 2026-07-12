package local

import "testing"

func TestExtractGemmaFinal(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "thought block then final channel",
			in:   "\n\n<|channel>thought\nThinking Process:\n...reasoning...\n5. Final Polish: X<channel|>Do the thing, but keep the old client.",
			want: "Do the thing, but keep the old client.",
		},
		{
			name: "real gemma output",
			in:   "\n\n<|channel>thought\n1. Analyze...\n(...)<channel|>Lama и Crisp конфликтуют каждый раунд, когда я их упоминаю, или они остаются в памяти для холодного старта?\n\n\n",
			want: "Lama и Crisp конфликтуют каждый раунд, когда я их упоминаю, или они остаются в памяти для холодного старта?",
		},
		{
			name: "no channel markers falls back to last line",
			in:   "some log\nRun tests then deploy.\n",
			want: "Run tests then deploy.",
		},
		{
			name: "empty",
			in:   "   \n  \n",
			want: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractGemmaFinal([]byte(c.in)); got != c.want {
				t.Fatalf("extractGemmaFinal() = %q, want %q", got, c.want)
			}
		})
	}
}
