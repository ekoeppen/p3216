use libc::{grantpt, posix_openpt, ptsname, unlockpt};
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::io::{Error, Write};
use std::mem;
use std::ops::Not;
use std::ops::{BitAnd, BitOr, BitXor};
use std::os::unix::io::{AsRawFd, FromRawFd};
use termios::*;

struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

const MEMORY_SIZE: usize = 2 * 1024 * 1024;

// Opcode 8, target 4, op1 4, op2 4, spare: 12
// Opcode 8, target 4, op1 4, imm 16

const FLAGS: usize = 13;
const LR: usize = 14;
const PC: usize = 15;

const ZERO: u32 = 1;
const NEGATIVE: u32 = 2;
const OVERFLOW: u32 = 4;

const VERSION: Version = Version {
    major: 1,
    minor: 6,
    patch: 1,
};

#[derive(Debug, Copy, Clone, PartialEq)]
enum Opcode {
    Undecoded = 0,
    Illegal = 1,
    NOP = 2,
    Move = 3,
    And = 4,
    Or = 5,
    Xor = 6,
    Not = 7,
    Add = 8,
    Sub = 9,
    Mul = 10,
    Div = 11,
    Load = 12,
    Store = 13,
    SetLow = 14,
    SetHigh = 15,
    RightShift = 16,
    LeftShift = 17,
    SignedDiv = 18,
    ArithmeticRightShift = 19,
    Syscall = 31,
}

enum Word {
    Low,
    High,
}

#[derive(Debug)]
enum Syscall {
    Putchar = 0,
    Getchar = 1,
    Quit = 2,
    MemDump = 3,
    VMDump = 4,
    DebugCtrl = 5,
    Open = 6,
    Close = 7,
    ReadByte = 8,
    WriteByte = 9,
    Argc = 10,
    Arg = 11,
    Create = 12,
    Write = 13,
    Read = 14,
    Version = 15,
    Pwd = 16,
    Cwd = 17,
    Grantpt = 18,
    Unlockpt = 19,
    Ptsname = 20,
    PosixOpenpt = 21,
    CfMakeRaw = 22,
    SetSpeed = 23,
}

#[derive(Debug)]
enum PrePostOps {
    None = 0,
    PreIncrement = 1,
    PostIncrement = 2,
    PreDecrement = 3,
    PostDecrement = 4,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Instruction {
    opcode: Opcode,
    target: usize,
    op1: usize,
    op2: usize,
    imm: u16,
    if_negative: bool,
    if_zero: bool,
    immediate: bool,
}

struct VM {
    debug: bool,
    registers: [u32; 16],
    memory: Vec<u8>,
    instruction_cache: Vec<Instruction>,
    args: Vec<String>,
}

fn print_vm(vm: &VM) {
    for r in 0..8 {
        print!("R{:02}: {:08x} ", r, vm.registers[r]);
    }
    println!();
    for r in 8..16 {
        print!("R{:02}: {:08x} ", r, vm.registers[r]);
    }
    println!();
}

fn print_instruction(instruction: Instruction) {
    print!(
        "{:?} T: {} Op1: {} ",
        instruction.opcode, instruction.target, instruction.op1
    );
    if (instruction.opcode == Opcode::Load || instruction.opcode == Opcode::Store)
        && !instruction.immediate
    {
        print!("{:?} ", pre_post_op(instruction.op2));
    } else {
        if instruction.immediate {
            print!("I: {:04x} ", instruction.imm);
        } else {
            print!("Op2: {} ", instruction.op2);
        }
    }
    if instruction.if_zero {
        print!("Z ");
    }
    if instruction.if_negative {
        print!("N ");
    }
    println!();
}

fn get_peripheral(_addr: usize) -> u32 {
    0
}

fn get_mem(vm: &VM, addr: usize) -> u32 {
    if addr >= 0xe0000000 {
        return get_peripheral(addr);
    }
    let r = vm.memory[addr] as u32
        + (256
            * (vm.memory[addr + 1] as u32
                + (256 * (vm.memory[addr + 2] as u32 + (256 * (vm.memory[addr + 3] as u32))))));
    if vm.debug {
        println!("[####] Get {:08x} -> {:08x}", addr, r);
    }
    r
}

fn set_mem(vm: &mut VM, addr: usize, value: u32) {
    vm.memory[addr] = (value & 0xff) as u8;
    vm.memory[addr + 1] = ((value >> 8) & 0xff) as u8;
    vm.memory[addr + 2] = ((value >> 16) & 0xff) as u8;
    vm.memory[addr + 3] = ((value >> 24) & 0xff) as u8;
    vm.instruction_cache[addr].opcode = Opcode::Undecoded;
    if vm.debug {
        println!("[####] Set {:08x} = {:08x}", addr, value);
    }
}

fn decode(bytes: &[u8]) -> Instruction {
    let if_zero: bool = (bytes[0] & 0x80) != 0;
    let if_negative: bool = (bytes[0] & 0x40) != 0;
    let immediate: bool = (bytes[0] & 0x20) != 0;
    let opcode_num = bytes[0] & 0x1f;
    let target = (bytes[1] as usize & 0xf0) >> 4;
    let op1 = bytes[1] as usize & 0x0f;
    let op2 = (bytes[2] as usize & 0xf0) >> 4;
    let imm = bytes[2] as u16 + (bytes[3] as u16) * 256;
    let opcode = match opcode_num {
        0 => Opcode::Undecoded,
        1 => Opcode::Illegal,
        2 => Opcode::NOP,
        3 => Opcode::Move,
        4 => Opcode::And,
        5 => Opcode::Or,
        6 => Opcode::Xor,
        7 => Opcode::Not,
        8 => Opcode::Add,
        9 => Opcode::Sub,
        10 => Opcode::Mul,
        11 => Opcode::Div,
        12 => Opcode::Load,
        13 => Opcode::Store,
        14 => Opcode::SetLow,
        15 => Opcode::SetHigh,
        16 => Opcode::RightShift,
        17 => Opcode::LeftShift,
        18 => Opcode::SignedDiv,
        19 => Opcode::ArithmeticRightShift,
        31 => Opcode::Syscall,
        _ => Opcode::Illegal,
    };
    Instruction {
        opcode,
        target,
        op1,
        op2,
        imm,
        if_negative,
        if_zero,
        immediate,
    }
}

fn pre_post_op(op: usize) -> PrePostOps {
    match op {
        1 => PrePostOps::PreIncrement,
        2 => PrePostOps::PostIncrement,
        3 => PrePostOps::PreDecrement,
        4 => PrePostOps::PostDecrement,
        _ => PrePostOps::None,
    }
}

fn syscall_number(num: u32) -> Syscall {
    match num {
        0 => Syscall::Getchar,
        1 => Syscall::Putchar,
        2 => Syscall::Quit,
        3 => Syscall::MemDump,
        4 => Syscall::VMDump,
        5 => Syscall::DebugCtrl,
        6 => Syscall::Open,
        7 => Syscall::Close,
        8 => Syscall::ReadByte,
        9 => Syscall::WriteByte,
        10 => Syscall::Argc,
        11 => Syscall::Arg,
        12 => Syscall::Create,
        13 => Syscall::Write,
        14 => Syscall::Read,
        15 => Syscall::Version,
        16 => Syscall::Pwd,
        17 => Syscall::Cwd,
        18 => Syscall::Grantpt,
        19 => Syscall::Unlockpt,
        20 => Syscall::Ptsname,
        21 => Syscall::PosixOpenpt,
        22 => Syscall::CfMakeRaw,
        23 => Syscall::SetSpeed,
        _ => Syscall::Quit,
    }
}

fn mem_dump(from: usize, length: usize, vm: &VM) {
    print!("{:08x}: ", from);
    for i in from..from + length {
        print!("{:02x} ", vm.memory[i]);
    }
    println!();
}

fn pre_op(instruction: Instruction, vm: &mut VM) {
    if instruction.immediate {
        return;
    }
    match pre_post_op(instruction.op2) {
        PrePostOps::PreIncrement => vm.registers[instruction.op1] += 4,
        PrePostOps::PreDecrement => vm.registers[instruction.op1] -= 4,
        _ => (),
    }
}

fn post_op(instruction: Instruction, vm: &mut VM) {
    if instruction.immediate {
        return;
    }
    match pre_post_op(instruction.op2) {
        PrePostOps::PostIncrement => vm.registers[instruction.op1] += 4,
        PrePostOps::PostDecrement => vm.registers[instruction.op1] -= 4,
        _ => (),
    }
}

fn load_store_addr(instruction: Instruction, vm: &VM) -> u32 {
    if !instruction.immediate {
        vm.registers[instruction.op1]
    } else if instruction.imm < 0x8000 {
        vm.registers[instruction.op1] + instruction.imm as u32
    } else {
        vm.registers[instruction.op1] - (0x10000 - instruction.imm as u32)
    }
}

fn set_flags(instruction: Instruction, vm: &mut VM) {
    let zero = vm.registers[instruction.target] == 0;
    let negative = vm.registers[instruction.target] > 0x7fffffff;
    vm.registers[FLAGS] = vm.registers[FLAGS] & !(ZERO | NEGATIVE)
        | if zero { ZERO } else { 0 }
        | if negative { NEGATIVE } else { 0 };
}

fn move_inst(instruction: Instruction, vm: &mut VM) {
    if instruction.target == PC {
        let old_pc = vm.registers[PC];
        vm.registers[PC] = vm.registers[instruction.op1];
        vm.registers[LR] = old_pc;
        if vm.debug {
            println!("[####] jump {:#010x}", vm.registers[PC]);
        }
        return;
    } else if instruction.immediate {
        vm.registers[instruction.target] = instruction.imm as u32;
    } else {
        vm.registers[instruction.target] = vm.registers[instruction.op1];
    }
    set_flags(instruction, vm);
}

fn load_inst(instruction: Instruction, vm: &mut VM) {
    pre_op(instruction, vm);
    let addr = load_store_addr(instruction, vm);
    if instruction.target == PC {
        let old_pc = vm.registers[PC];
        vm.registers[PC] = get_mem(vm, addr as usize);
        vm.registers[LR] = old_pc;
        if vm.debug {
            println!("[####] jump {:#010x}", vm.registers[PC]);
        }
    } else {
        vm.registers[instruction.target] = get_mem(vm, addr as usize);
        set_flags(instruction, vm);
    }
    post_op(instruction, vm);
}

fn store_inst(instruction: Instruction, vm: &mut VM) {
    pre_op(instruction, vm);
    let addr = load_store_addr(instruction, vm);
    set_mem(vm, addr as usize, vm.registers[instruction.target]);
    post_op(instruction, vm);
}

fn set_inst(instruction: Instruction, part: Word, vm: &mut VM) {
    let (val, mask): (u32, u32) = match part {
        Word::Low => (instruction.imm as u32, 0xffff0000),
        Word::High => ((instruction.imm as u32) << 16, 0x0000ffff),
    };
    vm.registers[instruction.target] = (vm.registers[instruction.target] & mask) | val;
    set_flags(instruction, vm);
}

fn apply_binary(instruction: Instruction, operation: fn(u32, u32) -> u32, vm: &mut VM) {
    if instruction.immediate {
        vm.registers[instruction.target] =
            operation(vm.registers[instruction.op1], instruction.imm as u32);
    } else {
        vm.registers[instruction.target] =
            operation(vm.registers[instruction.op1], vm.registers[instruction.op2]);
    }
    set_flags(instruction, vm);
}

fn apply_binary_overflow(
    instruction: Instruction,
    operation: fn(u32, u32) -> (u32, bool),
    vm: &mut VM,
) {
    let (result, overflow): (u32, bool) = if instruction.immediate {
        operation(vm.registers[instruction.op1], instruction.imm as u32)
    } else {
        operation(vm.registers[instruction.op1], vm.registers[instruction.op2])
    };
    vm.registers[instruction.target] = result;
    set_flags(instruction, vm);
    vm.registers[FLAGS] = vm.registers[FLAGS] & !OVERFLOW | if overflow { OVERFLOW } else { 0 };
}

fn apply_unary(instruction: Instruction, operation: fn(u32) -> u32, vm: &mut VM) {
    vm.registers[instruction.target] = operation(vm.registers[instruction.op1]);
    set_flags(instruction, vm);
}

fn syscall(instruction: Instruction, vm: &mut VM) {
    let s = if instruction.immediate {
        instruction.imm as u32
    } else {
        vm.registers[instruction.op2]
    };
    let _ = match syscall_number(s) {
        Syscall::Putchar => {
            std::io::stdout()
                .write(&[vm.registers[instruction.target] as u8])
                .unwrap();
            std::io::stdout().flush().unwrap();
        }
        Syscall::Getchar => {
            let mut b = [0];
            std::io::stdin().read(&mut b).unwrap();
            vm.registers[instruction.op1] = b[0] as u32;
        }
        Syscall::MemDump => {
            mem_dump(
                vm.registers[instruction.target] as usize,
                vm.registers[instruction.op1] as usize,
                vm,
            );
        }
        Syscall::VMDump => {
            print_vm(vm);
        }
        Syscall::Quit => std::process::exit(vm.registers[instruction.target] as i32),
        Syscall::DebugCtrl => vm.debug = vm.registers[instruction.target] != 0,
        Syscall::ReadByte => {
            let mut b = [0];
            let mut f = unsafe { File::from_raw_fd(vm.registers[instruction.target] as i32) };
            let n: i32 = match f.read(&mut b) {
                Ok(n) => {
                    if n > 0 {
                        b[0] as i32
                    } else {
                        -1
                    }
                }
                Err(_) => -1,
            };
            vm.registers[instruction.op1] = n as u32;
            mem::forget(f);
        }
        Syscall::WriteByte => {
            let mut f = unsafe { File::from_raw_fd(vm.registers[instruction.target] as i32) };
            let _ = f.write(&[vm.registers[instruction.op1] as u8]);
            mem::forget(f);
        }
        Syscall::Read => {
            let p = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let mut f = unsafe { File::from_raw_fd(vm.registers[instruction.op1 + 1] as i32) };
            match f.read(&mut vm.memory[p..p + n]) {
                Ok(n) => {
                    vm.registers[instruction.target] = 0;
                    vm.registers[instruction.op1] = n as u32;
                }
                Err(_) => {
                    vm.registers[instruction.target] = u32::MAX;
                }
            };
            mem::forget(f);
        }
        Syscall::Write => {
            let p = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let mut f = unsafe { File::from_raw_fd(vm.registers[instruction.op1 + 1] as i32) };
            match f.write(&vm.memory[p..p + n].to_vec()) {
                Ok(_) => {
                    vm.registers[instruction.target] = 0;
                    f.flush().unwrap();
                }
                Err(_) => {
                    vm.registers[instruction.target] = u32::MAX;
                }
            };
            mem::forget(f);
        }
        Syscall::Open => {
            let p = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let _options = vm.registers[instruction.op1 + 1] as usize;
            let path = String::from_utf8(vm.memory[p..p + n].to_vec()).unwrap();
            match File::options().read(true).write(true).open(path) {
                Ok(f) => {
                    vm.registers[instruction.target] = 0;
                    vm.registers[instruction.op1] = f.as_raw_fd() as u32;
                    mem::forget(f);
                }
                Err(_) => {
                    vm.registers[instruction.target] = u32::MAX;
                }
            };
        }
        Syscall::Create => {
            let p = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let path = String::from_utf8(vm.memory[p..p + n].to_vec()).unwrap();
            match File::create(path) {
                Ok(f) => {
                    vm.registers[instruction.target] = 0;
                    vm.registers[instruction.op1] = f.as_raw_fd() as u32;
                    mem::forget(f);
                }
                Err(_) => {
                    vm.registers[instruction.target] = u32::MAX;
                }
            };
        }
        Syscall::Close => {
            let _ = unsafe { File::from_raw_fd(vm.registers[instruction.target] as i32) };
            vm.registers[instruction.op1] = 0;
        }
        Syscall::Argc => {
            vm.registers[instruction.target] = vm.args.len() as u32;
        }
        Syscall::Arg => {
            let a = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let addr = vm.registers[instruction.op1 + 1] as usize;
            match vm.args.get(a) {
                Some(arg) => {
                    let r = if n < arg.len() { n } else { arg.len() };
                    vm.memory[addr..addr + r].copy_from_slice(&arg.as_bytes()[0..r]);
                    vm.registers[instruction.op1 + 1] = r as u32;
                }
                None => vm.registers[instruction.op1 + 1] = 0,
            }
        }
        Syscall::Pwd => {
            let addr = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let p = std::env::current_dir().unwrap();
            let path = p.to_str().unwrap();
            let r = if n < path.len() { n } else { path.len() };
            vm.memory[addr..addr + r].copy_from_slice(&path.as_bytes()[0..r]);
            vm.registers[instruction.target] = r as u32;
        }
        Syscall::Cwd => {
            let p = vm.registers[instruction.target] as usize;
            let n = vm.registers[instruction.op1] as usize;
            let path = String::from_utf8(vm.memory[p..p + n].to_vec()).unwrap();
            match std::env::set_current_dir(path) {
                Ok(f) => {
                    vm.registers[instruction.target] = 0;
                    let _ = f;
                }
                Err(_) => {
                    vm.registers[instruction.target] = u32::MAX;
                }
            };
        }
        Syscall::Version => {
            vm.registers[instruction.target] = VERSION.major;
            vm.registers[instruction.op1] = VERSION.minor;
            vm.registers[instruction.op1 + 1] = VERSION.patch;
        }
        Syscall::Grantpt => unsafe {
            vm.registers[instruction.op1] = grantpt(vm.registers[instruction.target] as i32) as u32;
        },
        Syscall::Unlockpt => unsafe {
            vm.registers[instruction.op1] =
                unlockpt(vm.registers[instruction.target] as i32) as u32;
        },
        Syscall::Ptsname => unsafe {
            let n = vm.registers[instruction.op1] as usize;
            let addr = vm.registers[instruction.op1 + 1] as usize;
            let c_str = ptsname(vm.registers[instruction.target] as i32);
            if c_str == std::ptr::null_mut() {
                vm.registers[instruction.target] = 0;
                return;
            }
            let name = CStr::from_ptr(c_str);
            let name_len = name.to_bytes().len();
            let r = if n < name_len { n } else { name_len };
            vm.memory[addr..addr + r].copy_from_slice(&name.to_bytes()[0..r]);
            vm.registers[instruction.target] = r as u32;
        },
        Syscall::PosixOpenpt => unsafe {
            let fd = posix_openpt(vm.registers[instruction.op1] as i32);
            vm.registers[instruction.target] = fd as u32;
        },
        Syscall::CfMakeRaw => {
            let fd = vm.registers[instruction.target] as i32;
            let mut attrs = Termios::from_fd(fd).unwrap();
            cfmakeraw(&mut attrs);
            match tcsetattr(fd, TCSANOW, &attrs) {
                Ok(_) => vm.registers[instruction.op1] = 0,
                Err(_) => vm.registers[instruction.op1] = u32::MAX,
            }
        }
        Syscall::SetSpeed => {
            let fd = vm.registers[instruction.target] as i32;
            let mut attrs = Termios::from_fd(fd).unwrap();

            #[cfg(target_os = "macos")]
            let speed = vm.registers[instruction.op1] as u64;
            #[cfg(target_os = "linux")]
            let speed = vm.registers[instruction.op1];

            cfsetispeed(&mut attrs, speed).unwrap();
            cfsetospeed(&mut attrs, speed).unwrap();
            match tcsetattr(fd, TCSANOW, &attrs) {
                Ok(_) => vm.registers[instruction.op1] = 0,
                Err(_) => vm.registers[instruction.op1] = u32::MAX,
            }
        }
    };
}

fn execute(vm: &mut VM) {
    loop {
        let pc = vm.registers[PC];
        vm.registers[PC] += 4;
        let pc_index = pc as usize;
        let instruction = vm.instruction_cache[pc_index];
        let zero = vm.registers[FLAGS] & ZERO != 0;
        let negative = vm.registers[FLAGS] & NEGATIVE != 0;
        if instruction.opcode == Opcode::Illegal {
            std::process::exit(1)
        }
        if instruction.if_negative && !negative {
            continue;
        }
        if instruction.if_zero && !zero {
            continue;
        }
        if instruction.opcode == Opcode::Undecoded {
            vm.instruction_cache[pc_index] = decode(&vm.memory[pc_index..pc_index + 4]);
            vm.registers[PC] -= 4;
            continue;
        }
        if vm.debug {
            print_vm(&vm);
            println!("{:#010x}: ", pc);
            print_instruction(instruction);
        }
        match instruction.opcode {
            Opcode::NOP => (),
            Opcode::Move => move_inst(instruction, vm),
            Opcode::And => apply_binary(instruction, u32::bitand, vm),
            Opcode::Or => apply_binary(instruction, u32::bitor, vm),
            Opcode::Xor => apply_binary(instruction, u32::bitxor, vm),
            Opcode::Not => apply_unary(instruction, u32::not, vm),
            Opcode::Add => apply_binary_overflow(instruction, u32::overflowing_add, vm),
            Opcode::Sub => apply_binary_overflow(instruction, u32::overflowing_sub, vm),
            Opcode::Mul => apply_binary_overflow(instruction, u32::overflowing_mul, vm),
            Opcode::Div => apply_binary_overflow(instruction, u32::overflowing_div, vm),
            Opcode::SignedDiv => apply_binary(
                instruction,
                |a, b| i32::wrapping_div(a as i32, b as i32) as u32,
                vm,
            ),
            Opcode::Load => load_inst(instruction, vm),
            Opcode::Store => store_inst(instruction, vm),
            Opcode::SetLow => set_inst(instruction, Word::Low, vm),
            Opcode::SetHigh => set_inst(instruction, Word::High, vm),
            Opcode::RightShift => apply_binary_overflow(instruction, u32::overflowing_shr, vm),
            Opcode::LeftShift => apply_binary_overflow(instruction, u32::overflowing_shl, vm),
            Opcode::Syscall => syscall(instruction, vm),
            Opcode::ArithmeticRightShift => apply_binary(
                instruction,
                |a, b| i32::wrapping_shr(a as i32, b) as u32,
                vm,
            ),
            Opcode::Illegal => (),
            Opcode::Undecoded => (),
        }
    }
}

fn init(vm: &mut VM) {
    for r in 0..16 {
        vm.registers[r] = get_mem(vm, r * 4);
    }
}

#[cfg(not(feature = "preload"))]
fn args() -> Vec<String> {
    let mut args = Vec::new();
    for arg in std::env::args() {
        args.push(arg);
    }
    return args;
}

#[cfg(feature = "preload")]
fn args() -> Vec<String> {
    let mut args = Vec::new();
    args.push(std::env::args().nth(0).unwrap());
    args.push("image.bin".to_string());
    for arg in std::env::args().skip(1) {
        args.push(arg);
    }
    return args;
}

fn run(image: Box<[u8; MEMORY_SIZE]>, debug: bool) -> std::io::Result<()> {
    let mut vm = VM {
        debug,
        registers: [0; 16],
        memory: Vec::new(),
        instruction_cache: Vec::new(),
        args: args(),
    };
    vm.memory.resize(MEMORY_SIZE, 0);
    vm.instruction_cache.resize(
        MEMORY_SIZE,
        Instruction {
            opcode: Opcode::Undecoded,
            if_negative: false,
            if_zero: false,
            target: 0,
            op1: 0,
            op2: 0,
            imm: 0,
            immediate: false,
        },
    );
    vm.memory[0..image.len()].copy_from_slice(&image[0..image.len()]);
    init(&mut vm);
    execute(&mut vm);
    Ok(())
}

#[cfg(not(feature = "preload"))]
fn load_image(name: Option<String>) -> std::io::Result<Box<[u8; MEMORY_SIZE]>> {
    match name {
        Some(name) => {
            let mut file = File::open(name)?;
            let mut image = Box::new([0; MEMORY_SIZE]);
            file.read(&mut *image)?;
            Ok(image)
        }
        None => {
            println!("Usage: p3212 <IMAGE>");
            return Err(Error::other("image argument missing"));
        }
    }
}

#[cfg(feature = "preload")]
fn load_image(_name: Option<String>) -> std::io::Result<Box<[u8; MEMORY_SIZE]>> {
    let preload = include_bytes!("../image.bin");
    let mut image = Box::new([0; MEMORY_SIZE]);
    image[..preload.len()].copy_from_slice(preload);
    Ok(image)
}

fn main() -> Result<(), std::io::Error> {
    let debug = false;
    match load_image(std::env::args().nth(1)) {
        Ok(image) => run(image, debug),
        Err(e) => {
            return Err(e);
        }
    }
}
