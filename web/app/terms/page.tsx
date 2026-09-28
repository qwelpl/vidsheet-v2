import type { Metadata } from "next";

export const metadata: Metadata = { title: "Terms of Service - Vidsheet" };

// NOTE for the operator: replace CONTACT_EMAIL with a real address you monitor
// for copyright / takedown notices.
const CONTACT_EMAIL = "abuse@vidsheet.app";

export default function TermsPage() {
  return (
    <div style={{ minHeight: "100vh", background: "var(--bg, #0a0b0e)", color: "var(--text, #e6e8ee)" }}>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "40px 22px 80px", lineHeight: 1.65, fontSize: 14 }}>
        <a href="/" style={{ color: "var(--accent)", fontSize: 13 }}>&larr; Back to Vidsheet</a>
        <h1 style={{ fontSize: 26, fontWeight: 800, marginTop: 18 }}>Terms of Service</h1>
        <p style={{ color: "var(--text-dim)" }}>
          Vidsheet transcribes Synthesia-style piano-roll videos into sheet music and MIDI. By
          using the service you agree to these terms.
        </p>

        <Section title="1. Your content and rights">
          You may only submit videos that you own, that are in the public domain, or that you
          otherwise have all necessary rights and permissions to upload, process, and create
          derivative works (such as sheet music and MIDI) from. You confirm this each time you
          queue a piece. You are solely responsible for the content you submit and for how you use
          the output.
        </Section>

        <Section title="2. Prohibited use">
          Do not upload or process copyrighted material you are not authorized to use. Do not use
          Vidsheet to infringe intellectual-property rights or to violate any law. We may refuse or
          remove any submission and suspend accounts that violate these terms.
        </Section>

        <Section title="3. Source videos and outputs">
          Source videos are deleted after processing and are not retained or published. Reconstructed
          outputs (sheet music, MIDI, and related files) are private to your account by default and
          are not added to any public library.
        </Section>

        <Section title="4. Copyright and takedown">
          If you believe content processed through Vidsheet infringes your copyright, send a notice
          to <a href={`mailto:${CONTACT_EMAIL}`} style={{ color: "var(--accent)" }}>{CONTACT_EMAIL}</a> that
          identifies the work, the material at issue, your contact information, and a good-faith
          statement of infringement. We will review and remove infringing material where
          appropriate.
        </Section>

        <Section title="5. Repeat infringers">
          Accounts that repeatedly submit infringing content, or that are the subject of repeated
          valid takedown notices, will have their access terminated.
        </Section>

        <Section title="6. No warranty; limitation of liability">
          The service is provided &ldquo;as is,&rdquo; without warranties of any kind. Transcription
          accuracy is not guaranteed. To the maximum extent permitted by law, Vidsheet and its
          operator are not liable for any damages arising from your use of the service or the
          output, including any claims related to content you submit.
        </Section>

        <Section title="7. Changes">
          These terms may be updated from time to time. Continued use of the service after a change
          constitutes acceptance of the updated terms.
        </Section>

        <p style={{ color: "var(--text-faint)", fontSize: 12, marginTop: 32 }}>
          Vidsheet is a transcription tool, not a music-licensing service. Obtaining any licenses
          required to use the output is your responsibility.
        </p>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginTop: 22 }}>
      <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>{title}</h2>
      <p style={{ color: "var(--text-dim)", margin: 0 }}>{children}</p>
    </div>
  );
}
