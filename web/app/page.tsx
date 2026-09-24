"use client";
import { useEffect, useRef, useState } from "react";
import { JobStatus, Project, getJob, getResult } from "@/lib/api";
import Landing from "./components/Landing";
import Progress from "./components/Progress";
import Workspace from "./components/Workspace";

export default function Page() {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    poll.current = setInterval(async () => {
      try {
        const s = await getJob(job.id);
        setJob(s);
        if (s.status === "done") {
          const r = await getResult(s.id);
          setProject(r);
        }
      } catch {
        /* transient */
      }
    }, 500);
    return () => { if (poll.current) clearInterval(poll.current); };
  }, [job?.id, job?.status]);

  function reset() {
    setJob(null);
    setProject(null);
    setProject(null);
  }

  if (!job) return <Landing onJob={setJob} />;
  if (job.status === "done" && project)
    return <Workspace jobId={job.id} project={project} onExit={reset} />;
  return <Progress job={job} onCancel={reset} />;
}
