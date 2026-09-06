//go:build windows

package local

import "syscall"

// childProcAttr is the Windows counterpart of the POSIX process group: there is
// no Setpgid, and referring to it at all breaks the build for GOOS=windows —
// which is not academic, the GPU node this supervisor is written for runs on
// Windows.
//
// CREATE_NEW_PROCESS_GROUP buys the same property that matters here: a console
// event delivered to inferenced is not propagated to the replicas, so they
// survive the parent exactly as they do on Unix.
const createNewProcessGroup = 0x00000200

func childProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: createNewProcessGroup}
}
