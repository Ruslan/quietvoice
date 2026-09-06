//go:build !windows

package local

import "syscall"

// childProcAttr puts a launched replica in its OWN process group.
//
// Why it matters: a replica is deliberately started with exec.Command, not
// exec.CommandContext, because it must outlive the admin request that started
// it — its lifecycle belongs to the supervisor, not to an HTTP handler. Its own
// group keeps a Ctrl-C (or any signal aimed at inferenced's group) from taking
// the replicas down with the parent as a side effect.
func childProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}
