# backend/manage_users.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable

from sqlmodel import SQLModel, Field, Session, select, create_engine
from passlib.context import CryptContext
from sqlalchemy import (
    String, Column, UniqueConstraint, ForeignKey, event
)
from sqlalchemy.engine import Engine

# ====== Config ======
BASE_DIR = Path(__file__).resolve().parent
DB_URL = f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}"

engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Ativa FKs no SQLite
@event.listens_for(Engine, "connect")
def _fk_on(dbapi_conn, conn_record):
    try:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    except Exception:
        pass

# ====== Models ======
class User(SQLModel, table=True):
    cpf: str = Field(primary_key=True, index=True)
    nome: str
    password_hash: str = Field(sa_column=Column("password", String, nullable=False))

class Invoice(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("cpf", "invoice_id", name="uq_invoice_per_user"),)
    id: int | None = Field(default=None, primary_key=True)
    cpf: str = Field(foreign_key="user.cpf", index=True)
    invoice_id: str = Field(index=True)

class PaymentIgnore(SQLModel, table=True):
    __table_args__ = (
        # evita criar duplicatas idênticas (ajuste se quiser outra chave)
        UniqueConstraint("patient_id", "date", "valor", "forma", "occurrences",
                         name="uq_payment_ignore"),
    )
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(index=True)      # ex.: 7314
    date: str | None = None                  # 'YYYY-MM-DD' (opcional)
    valor: float | None = None               # ex.: 90300.00 (opcional)
    forma: str | None = None                 # ex.: '8' (opcional)
    occurrences: int = 1                     # quantas ocorrências ignorar
    note: str | None = None                  # observação

def add_payment_ignore(patient_id: int, date: str | None = None,
                       valor: float | None = None, forma: str | None = None,
                       occurrences: int = 1, note: str | None = None) -> None:
    with Session(engine) as s:
        rec = PaymentIgnore(
            patient_id=patient_id,
            date=date,
            valor=valor,
            forma=str(forma) if forma is not None else None,
            occurrences=occurrences,
            note=note,
        )
        s.add(rec)
        s.commit()
        print(f"Exceção criada: #{rec.id} (patient_id={patient_id})")

def list_payment_ignores_cli(patient_id: int | None = None) -> None:
    with Session(engine) as s:
        stmt = select(PaymentIgnore)
        if patient_id:
            stmt = stmt.where(PaymentIgnore.patient_id == patient_id)
        rows = s.exec(stmt.order_by(PaymentIgnore.patient_id, PaymentIgnore.date)).all()
        if not rows:
            print("Nenhuma exceção cadastrada.")
            return
        for r in rows:
            print(f"- #{r.id} pid={r.patient_id} date={r.date or '—'} "
                  f"valor={r.valor if r.valor is not None else '—'} "
                  f"forma={r.forma or '—'} x{r.occurrences} note={r.note or ''}")

def del_payment_ignore(ignore_id: int) -> None:
    with Session(engine) as s:
        rec = s.get(PaymentIgnore, ignore_id)
        if not rec:
            print("Exceção não encontrada.")
            return
        s.delete(rec)
        s.commit()
        print(f"Exceção removida: #{ignore_id}")

# ====== Helpers ======
def so_digitos(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())

def hash_pwd(pwd: str) -> str:
    return pwd_ctx.hash(pwd)

def get_user(sess: Session, cpf: str) -> Optional[User]:
    return sess.get(User, cpf)

def parse_invoices(args_invoices: Optional[Iterable[str]]) -> list[str]:
    """
    Aceita:
    - várias flags: --invoice 123 --invoice 456
    - ou única flag com vírgula: --invoice 123,456
    """
    if not args_invoices:
        return []
    out: list[str] = []
    for raw in args_invoices:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out.extend(parts)
    # evita vazios e duplicados mantendo ordem
    seen = set()
    uniq = []
    for inv in out:
        if inv not in seen:
            uniq.append(inv)
            seen.add(inv)
    return uniq

# ====== Ações de usuário ======
def add_user(cpf: str, nome: str, senha: Optional[str] = None,
             overwrite: bool = False, invoices: Optional[list[str]] = None) -> None:
    cpf_n = so_digitos(cpf)
    if not cpf_n:
        print("CPF inválido.")
        return
    senha_final = so_digitos(senha) if senha else cpf_n  # padrão = CPF
    invoices = invoices or []

    with Session(engine) as s:
        u = get_user(s, cpf_n)
        if u and not overwrite:
            print(f"Já existe: {cpf_n}  (use --overwrite para atualizar)")
            return

        if u and overwrite:
            u.nome = nome or u.nome
            u.password_hash = hash_pwd(senha_final)
            s.add(u)
            # adiciona invoices novos (sem duplicar)
            for inv in invoices:
                try:
                    s.add(Invoice(cpf=cpf_n, invoice_id=inv))
                except Exception:
                    pass
            s.commit()
            print(f"Atualizado: {cpf_n} (senha redefinida) — invoices adicionados (se houver).")
        else:
            novo = User(cpf=cpf_n, nome=nome, password_hash=hash_pwd(senha_final))
            s.add(novo)
            s.commit()
            # adiciona invoices
            for inv in invoices:
                s.add(Invoice(cpf=cpf_n, invoice_id=inv))
            s.commit()
            print(f"Cadastrado: {cpf_n} (senha = CPF) — invoices: {', '.join(invoices) if invoices else 'nenhum'}")

def reset_password(cpf: str, nova_senha: Optional[str] = None) -> None:
    cpf_n = so_digitos(cpf)
    nova = so_digitos(nova_senha) if nova_senha else cpf_n  # padrão = CPF
    with Session(engine) as s:
        u = get_user(s, cpf_n)
        if not u:
            print("CPF não encontrado.")
            return
        u.password_hash = hash_pwd(nova)
        s.add(u)
        s.commit()
        msg = "CPF" if not nova_senha else "nova senha"
        print(f"Senha de {cpf_n} redefinida ({msg}).")

# manage_users.py  -> substitua a função list_users() inteira
def list_users():
    from sqlmodel import Session, select
    with Session(engine) as s:
        users = s.exec(select(User).order_by(User.cpf)).all()
        print("\nUsuários:")
        for u in users:
            invs = s.exec(
                select(Invoice.invoice_id).where(Invoice.cpf == u.cpf)
            ).all()

            # compatibilidade: pode vir como ["11719"] ou [("11719",)]
            if invs and isinstance(invs[0], tuple):
                inv_ids = [row[0] for row in invs]
            else:
                inv_ids = list(invs)

            inv_str = ", ".join(inv_ids) if inv_ids else "—"
            print(f"- {u.cpf} | {u.nome} | invoices: {inv_str}")

def delete_user(cpf: str) -> None:
    cpf_n = so_digitos(cpf)
    with Session(engine) as s:
        u = get_user(s, cpf_n)
        if not u:
            print("CPF não encontrado.")
            return
        # invoices com FK e ON DELETE RESTRICT por padrão;
        # apagamos explicitamente para manter simples:
        s.exec(select(Invoice).where(Invoice.cpf == cpf_n))
        s.query(Invoice).filter(Invoice.cpf == cpf_n).delete()  # type: ignore
        s.delete(u)
        s.commit()
        print(f"Removido: {cpf_n} (e seus invoices)")

# ====== Ações de invoices ======
def add_invoice(cpf: str, invoice_id: str) -> None:
    cpf_n = so_digitos(cpf)
    if not invoice_id:
        print("Informe o invoice_id.")
        return
    with Session(engine) as s:
        if not get_user(s, cpf_n):
            print("CPF não encontrado.")
            return
        try:
            s.add(Invoice(cpf=cpf_n, invoice_id=invoice_id))
            s.commit()
            print(f"Invoice '{invoice_id}' adicionado para {cpf_n}.")
        except Exception:
            print("Este invoice já existe para esse usuário.")

def list_invoices(cpf: str) -> None:
    cpf_n = so_digitos(cpf)
    with Session(engine) as s:
        if not get_user(s, cpf_n):
            print("CPF não encontrado.")
            return
        invs = s.exec(select(Invoice.id, Invoice.invoice_id).where(Invoice.cpf == cpf_n)).all()
        if not invs:
            print("Nenhum invoice para este usuário.")
            return
        print(f"Invoices de {cpf_n}:")
        for inv_id, inv in invs:
            print(f"- #{inv_id}  {inv}")

def del_invoice(cpf: str, invoice_id: str) -> None:
    cpf_n = so_digitos(cpf)
    with Session(engine) as s:
        q = s.exec(
            select(Invoice).where(Invoice.cpf == cpf_n, Invoice.invoice_id == invoice_id)
        ).first()
        if not q:
            print("Invoice não encontrado para este usuário.")
            return
        s.delete(q)
        s.commit()
        print(f"Invoice '{invoice_id}' removido de {cpf_n}.")

# ====== CLI ======
if __name__ == "__main__":
    import argparse
    SQLModel.metadata.create_all(engine)  # cria a tabela invoices se não existir

    p = argparse.ArgumentParser(description="Gerenciar usuários e invoices do Portal")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Cadastrar usuário (senha padrão = CPF)")
    a.add_argument("--cpf", required=True)
    a.add_argument("--nome", required=True)
    a.add_argument("--senha", help="Opcional; por padrão usa o CPF")
    a.add_argument("--overwrite", action="store_true", help="Atualiza se já existir")
    a.add_argument("--invoice", action="append",
                   help="Pode repetir a opção ou passar lista separada por vírgula")

    r = sub.add_parser("reset", help="Resetar senha para CPF (ou nova senha)")
    r.add_argument("--cpf", required=True)
    r.add_argument("--senha", help="Se omitido, usa o próprio CPF")

    l = sub.add_parser("list", help="Listar usuários (com invoices)")

    d = sub.add_parser("delete", help="Remover usuário")
    d.add_argument("--cpf", required=True)

    # exceptions
    exc = sub.add_parser("exceptions", help="Gerenciar exceções de pagamentos")
    exc_sub = exc.add_subparsers(dest="ecmd", required=True)

    exc_add = exc_sub.add_parser("add", help="Adicionar exceção")
    exc_add.add_argument("--patient_id", type=int, required=True)
    exc_add.add_argument("--date", help="YYYY-MM-DD")
    exc_add.add_argument("--valor", type=float)
    exc_add.add_argument("--forma")
    exc_add.add_argument("--occurrences", type=int, default=1)
    exc_add.add_argument("--note")

    exc_list = exc_sub.add_parser("list", help="Listar exceções")
    exc_list.add_argument("--patient_id", type=int)

    exc_del = exc_sub.add_parser("del", help="Excluir exceção por ID")
    exc_del.add_argument("--id", type=int, required=True)

    # invoices
    inv = sub.add_parser("invoices", help="Gerenciar invoices")
    inv_sub = inv.add_subparsers(dest="icmd", required=True)

    inv_add = inv_sub.add_parser("add", help="Adicionar invoice a um usuário")
    inv_add.add_argument("--cpf", required=True)
    inv_add.add_argument("--invoice", required=True)

    inv_list = inv_sub.add_parser("list", help="Listar invoices de um usuário")
    inv_list.add_argument("--cpf", required=True)

    inv_del = inv_sub.add_parser("del", help="Remover um invoice específico")
    inv_del.add_argument("--cpf", required=True)
    inv_del.add_argument("--invoice", required=True)

    args = p.parse_args()

    if args.cmd == "add":
        add_user(args.cpf, args.nome, args.senha, args.overwrite, parse_invoices(args.invoice))
    elif args.cmd == "reset":
        reset_password(args.cpf, args.senha)
    elif args.cmd == "list":
        list_users()
    elif args.cmd == "delete":
        delete_user(args.cpf)
    elif args.cmd == "invoices":
        if args.icmd == "add":
            add_invoice(args.cpf, args.invoice)
        elif args.icmd == "list":
            list_invoices(args.cpf)
        elif args.icmd == "del":
            del_invoice(args.cpf, args.invoice)
    elif args.cmd == "exceptions":
        if args.ecmd == "add":
            add_payment_ignore(args.patient_id, args.date, args.valor, args.forma,
                               args.occurrences, args.note)
        elif args.ecmd == "list":
            list_payment_ignores_cli(args.patient_id)
        elif args.ecmd == "del":
            del_payment_ignore(args.id)

