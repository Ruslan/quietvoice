package inference

import "testing"

func TestAgentText(t *testing.T) {
	cases := []struct {
		name              string
		intent, tone, urg string
		want              string
	}{
		{"calm-no-marker", "Let's ship it.", "neutral", "low", "Let's ship it."},
		{"empty-fields-no-marker", "Deploy now.", "", "", "Deploy now."},
		{"both-notable", "TURN IT OFF.", "frustrated", "high", "[urgency: high · tone: frustrated] TURN IT OFF."},
		{"tone-only", "(laughs) sure, whatever.", "amused", "low", "[tone: amused] (laughs) sure, whatever."},
		{"urgency-only", "Stop the deploy.", "neutral", "high", "[urgency: high] Stop the deploy."},
		{"case-insensitive-calm", "ok.", "Neutral", "Low", "ok."},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := &InterpretResult{Intent: c.intent, Tone: c.tone, Urgency: c.urg}
			if got := r.AgentText(); got != c.want {
				t.Errorf("AgentText() = %q, want %q", got, c.want)
			}
		})
	}
}
