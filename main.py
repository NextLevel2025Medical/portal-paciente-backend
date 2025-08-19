# backend/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Optional
import os
import httpx
from dotenv import load_dotenv
from datetime import date, timedelta, datetime
from calendar import monthrange
from sqlmodel import SQLModel, Field, Session, select, create_engine
from sqlalchemy import String, Column, text
from passlib.context import CryptContext
import re

SELLER_WA = {
    "Johnny":   "5531985252115",
    "Ana Maria":"553172631346",
    "Carolina": "553195426283",
}

# ------------------------------------------------------------------
# Carrega variáveis do .env (FEEGOW_BASE, FEEGOW_TOKEN, etc.)
# ------------------------------------------------------------------
load_dotenv()

app = FastAPI(title="Portal do Paciente - API")

origins = [
    "https://github.com/NextLevel2025Medical/portal-paciente-web",  # sua URL do web no Render
    "https://app.seudominio.com.br",                # seu subdomínio (quando apontar)
]

# CORS liberado no dev; em produção restrinja ao seu domínio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Users (SQLite + bcrypt) ======
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_pw(p: str) -> str:
    return pwd_ctx.hash(p)

def verify_pw(p: str, h: str) -> bool:
    return pwd_ctx.verify(p, h)

class User(SQLModel, table=True):
    cpf: str = Field(primary_key=True, index=True)
    nome: str
    # importante: mapear para a COLUNA "password" do banco
    password_hash: str = Field(sa_column=Column("password", String, nullable=False))
    vendedor: Optional[str] = Field(default=None)  # << NOVO
class Invoice(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    cpf: str
    invoice_id: str
class PaymentIgnore(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int                      # 7314
    date: Optional[str] = None           # '2025-08-05' (YYYY-MM-DD) – opcional
    valor: Optional[float] = None        # 90300.00 – opcional
    forma: Optional[str] = None          # '8' – opcional
    occurrences: int = 1                 # quantas ocorrências devemos ignorar
    note: Optional[str] = None           # observação

def list_invoice_ids_by_cpf(cpf: str) -> list[str]:
    """
    Retorna todos os invoice_id vinculados ao CPF na tabela invoice (sem duplicar e preservando ordem).
    """
    import re
    cpf_num = re.sub(r"\D+", "", cpf or "")
    if not cpf_num:
        return []
    with Session(engine) as s:
        rows = s.exec(select(Invoice.invoice_id).where(Invoice.cpf == cpf_num)).all()
        uniq: list[str] = []
        for r in rows:
            # Suporta formatos (valor direto, tupla, objeto)
            v = (r[0] if isinstance(r, (tuple, list)) else
                 (getattr(r, "invoice_id", None) if hasattr(r, "invoice_id") else r))
            v = str(v) if v is not None else ""
            if v and v not in uniq:
                uniq.append(v)
        return uniq

engine = create_engine("sqlite:///./app.db", connect_args={"check_same_thread": False})

def get_user(cpf: str) -> User | None:
    with Session(engine) as s:
        return s.get(User, cpf)

def create_user(cpf: str, nome: str, password: str):
    with Session(engine) as s:
        if s.get(User, cpf):
            return
        s.add(User(cpf=cpf, nome=nome, password_hash=hash_pw(password)))
        s.commit()

# --- util para adicionar coluna se faltar (SQLite) ---
def ensure_column(engine, table: str, column: str, sqltype: str):
    # funciona em SQLite; adapta para outros bancos se precisar
    with engine.connect() as con:
        cols = [row[1] for row in con.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            con.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {sqltype}")

@app.on_event("startup")
def _init_db():
    SQLModel.metadata.create_all(engine)
    # Seed opcional p/ desenvolvimento:
    ensure_column(engine, "user", "vendedor", "TEXT")
    create_user("12345678901", "Paciente Exemplo", "1234")

def list_payment_ignores(patient_id: int) -> list[PaymentIgnore]:
    with Session(engine) as s:
        return s.exec(select(PaymentIgnore).where(PaymentIgnore.patient_id == patient_id)).all()

def payment_matches_ignore(p: dict, ign: PaymentIgnore) -> bool:
    # p = {"data":"YYYY-MM-DD", "valor":float, "forma": "id ou texto", ...}
    if ign.date and str(p.get("data")) != ign.date:
        return False
    if ign.valor is not None and float(p.get("valor") or 0.0) != float(ign.valor):
        return False
    if ign.forma and str(p.get("forma")) != str(ign.forma):
        return False
    return True

def apply_payment_ignores(pagamentos: list[dict], patient_id: int) -> tuple[list[dict], list[dict]]:
    ignores = list_payment_ignores(patient_id)
    # contador por ignore.id para consumir só 'occurrences' vezes
    consumed: dict[int,int] = {ign.id: 0 for ign in ignores if ign.id is not None}
    applied: list[dict] = []

    out = []
    for p in pagamentos:
        ignored = False
        for ign in ignores:
            if payment_matches_ignore(p, ign):
                c = consumed.get(ign.id or -1, 0)
                if c < (ign.occurrences or 1):
                    consumed[ign.id] = c + 1
                    applied.append({"ignore_id": ign.id, "payment": p})
                    ignored = True
                    break
        if not ignored:
            out.append(p)
    return out, applied

# ------------------------------------------------------------------
# Cliente Feegow
# ------------------------------------------------------------------
class FeegowClient:
    def __init__(self) -> None:
        self.base = os.getenv("FEEGOW_BASE", "https://api.feegow.com/v1/api").rstrip("/")
        self.token = os.getenv("FEEGOW_TOKEN", "")
        self.auth_header = os.getenv("FEEGOW_AUTH_HEADER", "x-access-token")
        self.auth_scheme = os.getenv("FEEGOW_AUTH_SCHEME", "")  # "Bearer" ou vazio
        self.client = httpx.Client(timeout=20.0, trust_env=True)
        self._status_by_id: dict[int, str] = {}
        self._status_last_fetch: float = 0.0  # epoch seconds

    # ----------------- util -----------------
    def _headers(self) -> Dict[str, str]:
        if not self.token:
            return {}
        value = f"{self.auth_scheme} {self.token}".strip()
        return {self.auth_header: value}

    def _only_digits(self, s: str) -> str:
        return re.sub(r"\D+", "", s or "")

    def _norm_date(self, s: Optional[str]) -> Optional[str]:
        """Converte 'dd-mm-aaaa' → 'aaaa-mm-dd' (quando necessário)."""
        if not s or not isinstance(s, str):
            return None
        try:
            d, m, y = s.split("-")
            if len(y) == 4:
                return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        except Exception:
            pass
        return s

    # ----------------- Paciente -----------------
    def get_patient_name_by_id(self, patient_id: int) -> Optional[str]:
        url = f"{self.base}/patient/search"
        params = {"paciente_id": patient_id}
        r = self.client.get(url, params=params, headers=self._headers())
        r.raise_for_status()
        data: Any = r.json()

        if isinstance(data, dict):
            if "content" in data and isinstance(data["content"], dict):
                return data["content"].get("nome") or data["content"].get("name")
            for key in ("data", "results", "result"):
                if key in data and isinstance(data[key], list) and data[key]:
                    item = data[key][0]
                    return item.get("nome") or item.get("name")
                if key in data and isinstance(data[key], dict):
                    item = data[key]
                    return item.get("nome") or item.get("name")
            if "nome" in data:
                return data["nome"]
        return None

    def debug_patient_search(self, patient_id: int) -> Dict[str, Any]:
        url = f"{self.base}/patient/search"
        params = {"paciente_id": patient_id}
        headers = self._headers()
        redacted = {k: ("<set>" if v else "") for k, v in headers.items()}
        try:
            r = self.client.get(url, params=params, headers=headers)
            try:
                body = r.json()
            except Exception:
                body = r.text
            return {
                "request": {"url": url, "params": params, "headers": redacted},
                "response": {"status_code": r.status_code, "body": body},
            }
        except Exception as e:
            return {"request": {"url": url, "params": params, "headers": redacted}, "error": str(e)}

    def _extract_patient_from_search(self, data) -> tuple[int | None, str | None]:
        """
        Tenta extrair (id, nome) do retorno do /patient/search,
        lidando com variações de estrutura.
        """
        if not isinstance(data, dict):
            return None, None

        content = data.get("content", data)

        # 1) dict simples
        if isinstance(content, dict):
            pid = (content.get("id")
                   or content.get("paciente_id")
                   or content.get("patient_id"))
            nome = (content.get("nome")
                    or content.get("name"))
            try:
                return (int(pid), nome) if pid is not None else (None, nome)
            except Exception:
                return None, nome

        # 2) lista de objetos
        if isinstance(content, list) and content:
            item = content[0]
            if isinstance(item, dict):
                pid = (item.get("id")
                       or item.get("paciente_id")
                       or item.get("patient_id"))
                nome = (item.get("nome")
                        or item.get("name"))
                try:
                    return (int(pid), nome) if pid is not None else (None, nome)
                except Exception:
                    return None, nome

        # 3) outras variações comuns
        for key in ("data", "results", "result"):
            val = data.get(key)
            if isinstance(val, list) and val:
                item = val[0]
                if isinstance(item, dict):
                    pid = (item.get("id")
                           or item.get("paciente_id")
                           or item.get("patient_id"))
                    nome = (item.get("nome")
                            or item.get("name"))
                    try:
                        return (int(pid), nome) if pid is not None else (None, nome)
                    except Exception:
                        return None, nome
            if isinstance(val, dict):
                pid = (val.get("id")
                       or val.get("paciente_id")
                       or val.get("patient_id"))
                nome = (val.get("nome")
                        or val.get("name"))
                try:
                    return (int(pid), nome) if pid is not None else (None, nome)
                except Exception:
                    return None, nome

        return None, None

    def get_patient_id_by_cpf(self, cpf: str) -> int:
        """
        Busca o paciente pelo CPF e retorna o ID.
        Tenta primeiro via query string (padrão), depois GET com JSON no corpo
        (alguns ambientes Feegow aceitam isso).
        """
        cpf_num = self._only_digits(cpf)
        if not cpf_num:
            return 0

        url = f"{self.base}/patient/search"

        # 1) tentativa por query string
        try:
            r = self.client.get(url, params={"paciente_cpf": cpf_num}, headers=self._headers())
            r.raise_for_status()
            pid, _ = self._extract_patient_from_search(r.json())
            if pid:
                return pid
        except Exception:
            pass

        # 2) fallback: GET com JSON no corpo (não usual, mas alguns exemplos usam)
        try:
            r = self.client.get(url, headers=self._headers(), json={"paciente_cpf": cpf_num})
            r.raise_for_status()
            pid, _ = self._extract_patient_from_search(r.json())
            if pid:
                return pid
        except Exception:
            pass

        return 0

    # por enquanto mantemos estático; depois trocamos por busca real
    def get_patient_by_cpf(self, cpf: str) -> tuple[int | None, str | None]:
        """Retorna (patient_id, nome) pelo CPF (combina as tentativas acima)."""
        cpf_num = self._only_digits(cpf)
        if not cpf_num:
            return None, None
        url = f"{self.base}/patient/search"
        # query
        try:
            r = self.client.get(url, params={"paciente_cpf": cpf_num}, headers=self._headers())
            r.raise_for_status()
            pid, nome = self._extract_patient_from_search(r.json())
            if pid:
                return pid, nome
        except Exception:
            pass
        # fallback json body
        try:
            r = self.client.get(url, headers=self._headers(), json={"paciente_cpf": cpf_num})
            r.raise_for_status()
            pid, nome = self._extract_patient_from_search(r.json())
            if pid:
                return pid, nome
        except Exception:
            pass
        return None, None

    # ----------------- Propostas -----------------
    def _try_get(self, url: str, params: dict):
        r = self.client.get(url, params=params, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def get_proposals_by_patient(self, patient_id: int, status_filter: Optional[str] = None):
        url = f"{self.base}/proposal/list"
        try:
            data = self._try_get(url, {"PacienteID": patient_id})
        except Exception:
            data = self._try_get(url, {"paciente_id": patient_id})

        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, list):
            return []

        proposals = []
        target = (status_filter or "").strip().lower()

        for p in content:
            # pega o texto do status em qualquer variação comum
            status_txt = (
                p.get("status")
                or p.get("status_nome")
                or p.get("status_name")
                or p.get("situacao")
                or p.get("proposal_status")
                or ""
            )
            status_norm = str(status_txt).strip().lower()

            # se pediram filtro e não bateu, pula
            if target and target not in status_norm:
                continue

            valor = p.get("value", 0) or 0
            itens = []
            proc = p.get("procedimentos", {})
            rows = proc.get("data") if isinstance(proc, dict) else None
            if isinstance(rows, list):
                for it in rows:
                    itens.append({
                        "nome": it.get("nome", ""),
                        "valor": float(it.get("valor", 0) or 0),
                    })

            raw_date = (
                p.get("proposal_date")
                or p.get("date")
                or p.get("data")
                or p.get("created_at")
                or p.get("dt_proposta")
            )

            proposals.append({
                "proposal_id": p.get("proposal_id"),
                "proposal_date": self._norm_date(raw_date),
                "valor": float(valor),
                "itens": itens,
                "status": status_txt,          # (útil para depurar/exibir, se quiser)
            })
        return proposals

    # ----------------- Traduções de IDs -----------------
    _proc_cache: Dict[int, str] = {}
    _prof_cache: Dict[int, str] = {}
    _status_cache: Dict[int, str] = {}

    def get_procedure_name(self, procedure_id: int) -> Optional[str]:
        if not procedure_id:
            return None
        # cache por ID
        if procedure_id in self._proc_cache:
            return self._proc_cache[procedure_id]

        url = f"{self.base}/procedures/list"
        r = self.client.get(
            url,
            params={"procedure_id": procedure_id},  # se a API ignorar, tratamos abaixo
            headers=self._headers()
        )
        r.raise_for_status()
        data = r.json()

        nm: Optional[str] = None
        if isinstance(data, dict):
            c = data.get("content")

            # quando volta um único objeto:
            if isinstance(c, dict):
                nm = c.get("nome") or c.get("procedure") or c.get("name")

            # quando volta uma lista de objetos:
            elif isinstance(c, list) and c:
                alvo = None
                for item in c:
                    try:
                        pid = (
                            item.get("procedimento_id")
                            or item.get("procedure_id")
                            or item.get("id")
                        )
                        if pid is not None and int(pid) == int(procedure_id):
                            alvo = item
                            break
                    except Exception:
                        continue
                # se achou o item correto, usa o nome dele;
                # senão, pelo menos não quebra (usa o primeiro como fallback)
                base = alvo if alvo is not None else c[0]
                nm = base.get("nome") or base.get("procedure") or base.get("name")

        if nm:
            self._proc_cache[procedure_id] = nm
        return nm

    def get_professional_name(self, professional_id: int) -> Optional[str]:
        if not professional_id:
            return None
        if professional_id in self._prof_cache:
            return self._prof_cache[professional_id]

        url = f"{self.base}/professional/list"
        r = self.client.get(url, params={"professional_id": professional_id}, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        nm: Optional[str] = None
        if isinstance(data, dict):
            c = data.get("content")

            if isinstance(c, dict):
                nm = c.get("nome") or c.get("name")
            elif isinstance(c, list) and c:
                alvo = None
                for item in c:
                    try:
                        pid = (
                            item.get("profissional_id")
                            or item.get("professional_id")
                            or item.get("id")
                        )
                        if pid is not None and int(pid) == int(professional_id):
                            alvo = item
                            break
                    except Exception:
                        continue
                base = alvo if alvo is not None else c[0]  # fallback para não quebrar
                nm = base.get("nome") or base.get("name")

        if nm:
            self._prof_cache[professional_id] = nm
        return nm

    def _ensure_status_map(self, max_age_seconds: int = 3600):
        """
        Garante que o mapa de status esteja carregado da API e não esteja velho.
        Por padrão, refresca a cada 1h.
        """
        import time
        now = time.time()
        if self._status_by_id and (now - self._status_last_fetch) < max_age_seconds:
            return  # cache ainda válido

        try:
            url = f"{self.base}/appoints/status"
            r = self.client.get(url, headers=self._headers(), timeout=30)
            r.raise_for_status()
            data = r.json()

            mapping: dict[int, str] = {}
            content = (data or {}).get("content")
            if isinstance(content, list):
                for it in content:
                    try:
                        sid = int(it.get("status_id") or it.get("id") or 0)
                        name = (it.get("nome") or it.get("name") or it.get("status") or "").strip()
                        if sid and name:
                            mapping[sid] = name
                    except Exception:
                        pass

            if mapping:
                self._status_by_id = mapping
                self._status_last_fetch = now
        except Exception:
            # se der erro, mantém o cache anterior (se houver) e segue a vida
            pass

    def get_status_name(self, status_id: int) -> str | None:
        """
        Retorna o nome do status compatível com o Feegow.
        Tenta via cache/API; se não conseguir, devolve algo genérico.
        """
        if not status_id:
            return None
        self._ensure_status_map()
        name = self._status_by_id.get(int(status_id))
        if name:
            return name
        # fallback ultra genérico, apenas para não quebrar UI
        return f"Status #{int(status_id)}"

    # ----------------- Agenda (janelas simples) -----------------
    def get_appointments_window(self, patient_id: int, days_before: int = 90, days_after: int = 90) -> List[Dict[str, Any]]:
        today = date.today()
        start = today - timedelta(days=abs(days_before))
        end = today + timedelta(days=abs(days_after))

        # Garanta <= 180 dias (Feegow reclama se ultrapassar)
        while (end - start).days > 180:
            end -= timedelta(days=1)

        fmt = lambda dt: dt.strftime("%d-%m-%Y")
        params = {"data_start": fmt(start), "data_end": fmt(end), "paciente_id": patient_id}

        url = f"{self.base}/appoints/search"
        r = self.client.get(url, params=params, headers=self._headers())
        r.raise_for_status()
        data = r.json()

        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, list):
            return []

        appts = []
        for a in content:
            appts.append({
                "agendamento_id": a.get("agendamento_id"),
                "data": self._norm_date(a.get("data") or a.get("date")),
                "hora": a.get("hora") or a.get("hour") or a.get("time"),
                "horario": a.get("horario") or a.get("hora") or a.get("hour") or a.get("time"),
                "procedimento_id": a.get("procedimento_id") or a.get("procedure_id"),
                "status_id": a.get("status_id"),
                "profissional_id": a.get("profissional_id") or a.get("professional_id"),
            })
        return appts

    # ----------------- Agenda (período arbitrário costurado) -----------------
    def _feegow_search(self, patient_id: int, start: date, end: date) -> List[Dict[str, Any]]:
        """Chama /appoints/search para [start, end] (<= 180 dias) e normaliza a saída."""
        fmt = lambda dt: dt.strftime("%d-%m-%Y")
        params = {"data_start": fmt(start), "data_end": fmt(end), "paciente_id": patient_id}
        url = f"{self.base}/appoints/search"
        r = self.client.get(url, params=params, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, list):
            return []
        out: List[Dict[str, Any]] = []
        for a in content:
            out.append({
                "agendamento_id": a.get("agendamento_id"),
                "data": self._norm_date(a.get("data") or a.get("date")),
                "hora": a.get("hora") or a.get("hour") or a.get("time"),
                "horario": a.get("horario") or a.get("hora") or a.get("hour") or a.get("time"),
                "procedimento_id": a.get("procedimento_id") or a.get("procedure_id"),
                "status_id": a.get("status_id"),
                "profissional_id": a.get("profissional_id") or a.get("professional_id"),
                
                # >>> NOVIDADE: carregar o nome cru que a Feegow já retorna na busca
                "profissional_nome_raw": (
                    a.get("profissional_nome")
                    or a.get("professional_name")
                    or a.get("profissional")
                    or a.get("professional")
                ),
                # (se a API já trouxer o nome do procedimento, é bom preservar também)
                "procedimento_nome_raw": (
                    a.get("procedimento_nome")
                    or a.get("procedure_name")
                    or a.get("procedimento")
                    or a.get("procedure")
                ),
            })
        return out

    def _get_status_map(self):
        """
        Busca e cacheia o mapa {status_id: nome} de /v1/api/appoints/status.
        """
        if "status_map" in self.cache:
            return self.cache["status_map"]

        url = f"{self.base}/v1/api/appoints/status"
        r = requests.get(url, headers=self._headers())
        data = r.json() if r.content else {}

        status_map = {}
        if isinstance(data, dict) and data.get("success") and isinstance(data.get("content"), list):
            for s in data["content"]:
                # tenta id/nome nos campos mais comuns
                sid = s.get("status_id") or s.get("id") or s.get("status") or s.get("codigo")
                nome = s.get("nome") or s.get("name") or s.get("descricao") or s.get("description")
                if sid is not None and nome:
                    status_map[int(sid)] = str(nome)

        self.cache["status_map"] = status_map
        return status_map

    def get_appointments_range(self, patient_id: int, start: date, end: date) -> Dict[str, Any]:
        """
        Costura várias janelas de até 180 dias entre 'start' e 'end'.
        Retorna {'items': [...], 'windows': [(start,end), ...]}.
        """
        if end < start:
            start, end = end, start

        max_span = 180  # dias
        items: Dict[Any, Dict[str, Any]] = {}  # dedupe por agendamento_id
        windows: List[Dict[str, str]] = []

        cur = start
        while cur <= end:
            win_end = min(cur + timedelta(days=max_span - 1), end)
            windows.append({"start": cur.isoformat(), "end": win_end.isoformat()})
            chunk = self._feegow_search(patient_id, cur, win_end)
            for ap in chunk:
                key = ap.get("agendamento_id") or (ap.get("data"), ap.get("horario"), ap.get("procedimento_id"))
                items[key] = ap
            cur = win_end + timedelta(days=1)

        return {"items": list(items.values()), "windows": windows}

    def _is_generic_prof_name(self, name: str | None) -> bool:
        if not name:
            return True
        n = name.strip().upper()
        # ajuste a lista conforme sua base
        return (
            n in {"ADMINISTRATIVO", "ADMINISTRATIVA", "RECEPÇÃO", "RECEPCAO"}
            or "ADMIN" in n
        )

    def _hydrate_appointments(self, appts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not appts:
            return []

        proc_ids   = sorted({a.get("procedimento_id")  for a in appts if a.get("procedimento_id")})
        prof_ids   = sorted({a.get("profissional_id")  for a in appts if a.get("profissional_id")})
        status_ids = sorted({a.get("status_id")        for a in appts if a.get("status_id")})

        proc_map   = {pid: self.get_procedure_name(pid)    for pid in proc_ids}
        prof_map   = {pid: self.get_professional_name(pid) for pid in prof_ids}
        status_map = {sid: self.get_status_name(sid)       for sid in status_ids}

        hydrated: List[Dict[str, Any]] = []
        for a in appts:
            data_raw = (a.get("data") or "").strip()
            data_iso = data_raw
            try:
                d, m, y = data_raw.split("-")
                data_iso = f"{y}-{m}-{d}"
            except Exception:
                pass

            proc_id = a.get("procedimento_id")
            prof_id = a.get("profissional_id")
            st_id   = a.get("status_id")

            # nomes vindos de mapas (traduções por ID)
            proc_name_from_id = proc_map.get(proc_id)
            prof_name_from_id = prof_map.get(prof_id)

            # nomes crus vindos da própria busca de agendamentos
            proc_name_raw = a.get("procedimento_nome_raw")
            prof_name_raw = a.get("profissional_nome_raw")

            # regra: preferir nome cru se existir e não for genérico
            prof_final = None
            if prof_name_raw and not self._is_generic_prof_name(prof_name_raw):
                prof_final = prof_name_raw
            elif prof_name_from_id and not self._is_generic_prof_name(prof_name_from_id):
                prof_final = prof_name_from_id
            # 2) se só houver genéricos, mostre o que existir (para não ficar em branco)
            if not prof_final:
                prof_final = prof_name_raw or prof_name_from_id

            # nome do procedimento: bruto > por ID > fallback
            proc_final = proc_name_raw or proc_name_from_id or "Procedimento"

            hydrated.append({
                "agendamento_id": a.get("agendamento_id"),
                "data": a.get("data"),
                "horario": a.get("horario") or a.get("hora"),
                "procedimento_id": proc_id,
                "procedimento_nome": proc_final or "Procedimento",
                "status_id": st_id,
                "status_nome": status_map.get(st_id),
                "profissional_id": prof_id,
                "profissional_nome": prof_final,  # <—— agora deve vir "Gustavo Aquino" quando disponível
                "status": status_map.get(st_id, f"#{st_id}"),
            })
        return hydrated

    def get_appointments_range_hydrated(self, patient_id: int, start: date, end: date) -> Dict[str, Any]:
        stitched = self.get_appointments_range(patient_id, start, end)
        stitched["items"] = self._hydrate_appointments(stitched["items"])
        return stitched
    

    # ----------------- debug helpers -----------------
    def debug_appoints(self, patient_id: int, days_before: int = 90, days_after: int = 90):
        today = date.today()
        start = today - timedelta(days=abs(days_before))
        end = today + timedelta(days=abs(days_after))
        while (end - start).days > 180:
            end -= timedelta(days=1)
        fmt = lambda dt: dt.strftime("%d-%m-%Y")
        params = {"data_start": fmt(start), "data_end": fmt(end), "paciente_id": patient_id}
        url = f"{self.base}/appoints/search"
        headers = self._headers()
        redacted = {k: ("<set>" if v else "") for k, v in headers.items()}
        try:
            r = self.client.get(url, params=params, headers=headers)
            try:
                body = r.json()
            except Exception:
                body = r.text
            return {"request": {"url": url, "params": params, "headers": redacted},
                    "response": {"status_code": r.status_code, "body": body}}
        except Exception as e:
            return {"request": {"url": url, "params": params, "headers": redacted},
                    "error": str(e)}
    
    
    def get_invoice_payments(
        self,
        invoice_id: str,
        data_start: str = "01-01-2000",
        data_end: str   = "31-08-2050",
        tipo_transacao: str = "C",
    ) -> list[dict]:
        url = f"{self.base}/financial/list-invoice"
        params = {
            "data_start": data_start,
            "data_end": data_end,
            "tipo_transacao": tipo_transacao,
            "invoice_id": str(invoice_id),
        }
        r = self.client.get(url, params=params, headers=self._headers(), timeout=30)
        r.raise_for_status()
        raw = r.json()

        # ---- procura robusta pela lista de pagamentos ----
        rows = []

        def deep_find_pagamentos(obj):
            # retorna a primeira lista sob a chave 'pagamentos' encontrada em qualquer nível
            if isinstance(obj, dict):
                if isinstance(obj.get("pagamentos"), list):
                    return obj["pagamentos"]
                for v in obj.values():
                    found = deep_find_pagamentos(v)
                    if isinstance(found, list):
                        return found
            elif isinstance(obj, list):
                # às vezes vem content=[{pagamentos:[...]}]
                for v in obj:
                    found = deep_find_pagamentos(v)
                    if isinstance(found, list):
                        return found
            return None

        # 1) tenta achar 'pagamentos' em qualquer nível
        rows = deep_find_pagamentos(raw) or []

        # 2) fallbacks comuns (alguns ambientes retornam direto uma lista)
        if not rows:
            if isinstance(raw, list):
                rows = raw
            elif isinstance(raw, dict):
                # content ou data podem ser listas ou dicts
                c = raw.get("content")
                d = raw.get("data")
                if isinstance(c, list):
                    rows = c
                elif isinstance(d, list):
                    rows = d
                elif isinstance(d, dict):
                    rows = d.get("pagamentos") if isinstance(d.get("pagamentos"), list) else []

        out: list[dict] = []
        for it in rows or []:
            if not isinstance(it, dict):
                continue

            dt_raw = (
                it.get("data_pagamento")
                or it.get("data")
                or it.get("dt_pagamento")
                or it.get("payment_date")
            )
            dt_norm = self._norm_date(str(dt_raw)) if dt_raw else None

            valor = self._money_to_float(
                it.get("valor_pago")
                or it.get("valor")
                or it.get("vl_pago")
                or it.get("amount")
            )

            forma = (
                it.get("forma_pagamento_nome")
                or it.get("forma_pagamento")
                or it.get("forma")
                or it.get("payment_method")
                or "—"
            )

            if dt_norm and valor:
                out.append({"data": dt_norm, "valor": valor, "forma": str(forma)})

        out.sort(key=lambda x: x["data"])
        return out

    def _money_to_float(self, valor) -> float:
        if valor is None:
            return 0.0
        # já é número?
        if isinstance(valor, (int, float)):
            # heurística: inteiro grande e múltiplo de 100 => centavos
            if isinstance(valor, int) and valor >= 100_000 and valor % 100 == 0:
                return valor / 100.0
            # float muito grande (veio como 942000.0) também
            if isinstance(valor, float) and valor >= 100_000:
                return valor / 100.0
            return float(valor)

        s = str(valor).strip()
        if s == "":
            return 0.0

        # só dígitos? pode ser centavos (ex.: 942000)
        if s.isdigit():
            n = int(s)
            if n >= 100_000 and n % 100 == 0:
                return n / 100.0
            return float(n)

        # formatos com milhar/decimal brasileiros
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0
    def debug_list_invoice(self, invoice_id: str,
                            data_start: str = "01-01-2000",
                            data_end: str   = "31-08-2050",
                            tipo_transacao: str = "C") -> dict:
        url = f"{self.base}/financial/list-invoice"
        params = {
            "data_start": data_start,
            "data_end": data_end,
            "tipo_transacao": tipo_transacao,
            "invoice_id": str(invoice_id),
        }
        headers = self._headers()
        redacted = {k: ("<set>" if v else "") for k, v in headers.items()}
        try:
            r = self.client.get(url, params=params, headers=headers, timeout=30)
            try:
                body = r.json()
            except Exception:
                body = r.text
            return {
                "request": {"url": url, "params": params, "headers": redacted},
                "response": {"status_code": r.status_code, "body": body},
            }
        except Exception as e:
            return {"request": {"url": url, "params": params, "headers": redacted},
                    "error": str(e)}

feegow = FeegowClient()
# ------------------------------------------------------------------
# Rotas principais
# ------------------------------------------------------------------
class LoginDTO(BaseModel):
    cpf: str
    password: str

@app.post("/auth/login")
def login(body: LoginDTO):
    user = get_user(body.cpf)
    if not user or not verify_pw(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    pid, nome_feegow = feegow.get_patient_by_cpf(body.cpf)
    if not pid:
        raise HTTPException(status_code=404, detail="CPF não encontrado")

    return {
        "patient_id": pid,
        "name": nome_feegow or user.nome,
        "cpf": body.cpf,
        "vendedor": user.vendedor or "",            # <— NOVO
        "wa_number": SELLER_WA.get(user.vendedor)   # <— OPCIONAL
    }

@app.get("/patient/{patient_id}/summary")
def summary(
    patient_id: int, 
    debug: int = Query(0),
    invoice_id: Optional[str] = Query(None, description="Invoice ID para buscar pagamentos"),
    cpf: Optional[str] = Query(None, description="CPF do usuário para buscar todos os invoice_id no app.db"),
    pay_start: str = Query("01-01-2000"),
    pay_end:   str = Query("31-08-2050"),
):
    """
    Monta o resumo. Propostas vêm do Feegow (proposal/list).
    Pagamentos/agendamentos permanecem stub em parte.
    """
    # Propostas reais
    propostas = []
    try:
        propostas = feegow.get_proposals_by_patient(patient_id,status_filter="executada")
    except httpx.HTTPError as e:
        if debug:
            print("Erro Feegow proposal/list:", e)

    data: Dict[str, Any] = {
        "paciente": {"id": patient_id, "nome": "Paciente Exemplo", "cpf": "—"},
        "propostas": propostas,
        "pagamentos": [
        ],
        "agendamentos": [
        ],
        "adicionais_sugeridos": [],
    }

    # >>> PAGAMENTOS: buscar todos os invoices do CPF (mais o invoice_id da URL, se vier) <<<
    invoice_ids: list[str] = []
    try:
        if cpf:
            invoice_ids.extend(list_invoice_ids_by_cpf(cpf))
        if invoice_id:
            iid = str(invoice_id)
            if iid not in invoice_ids:
                invoice_ids.append(iid)
            try:
                pagos = feegow.get_invoice_payments(
                    invoice_id=invoice_id,
                    data_start=pay_start,
                    data_end=pay_end,
                    tipo_transacao="C",
                )
                if debug:
                    data.setdefault("_debug", {})["feegow_list_invoice"] = {
                        "invoice_id": str(invoice_id),
                        "params": {"data_start": pay_start, "data_end": pay_end, "tipo_transacao": "C"},
                        "qtd": len(pagos or []),
                    }
                if isinstance(pagos, list) and pagos:
                    data["pagamentos"] = pagos
            except httpx.HTTPError as e:
                if debug:
                    data.setdefault("_debug", {})["pagamentos_error"] = {
                        "error": str(e),
                        "inspect": feegow.debug_list_invoice(
                            invoice_id=invoice_id,
                            data_start=pay_start,
                            data_end=pay_end,
                            tipo_transacao="C",
                        ),
                    }
    except Exception as e:
        if debug:
            data.setdefault("_debug", {})["invoice_query_error"] = str(e)

    if invoice_ids:
        try:
            todos = []
            for inv in invoice_ids:
                try:
                    pagos = feegow.get_invoice_payments(
                        invoice_id=inv,
                        data_start=pay_start,
                        data_end=pay_end,
                        tipo_transacao="C",
                    )
                    todos.extend(pagos or [])
                except httpx.HTTPError as ee:
                    if debug:
                        data.setdefault("_debug", {}).setdefault("pagamentos_errors", {})[inv] = str(ee)

            # ordena por data e aplica (se nada vier, mantém stub)
            if todos:
                todos.sort(key=lambda x: x.get("data") or "")
                data["pagamentos"] = todos

            # útil p/ conferir na UI durante dev
            if debug:
                data.setdefault("_debug", {})["invoice_ids_usados"] = invoice_ids
        except Exception as e:
            if debug:
                data.setdefault("_debug", {})["pagamentos_error"] = str(e)

    # Nome real do paciente (se disponível)
    try:
        nome_real = feegow.get_patient_name_by_id(patient_id)
        if nome_real:
            data["paciente"]["nome"] = nome_real
    except httpx.HTTPError:
        pass
    
    # --- APLICA EXCEÇÕES DE PAGAMENTOS (se houver para este paciente) ---
    raw_pagamentos = list(data.get("pagamentos") or [])
    pagamentos_filtrados, ignores_aplicados = apply_payment_ignores(raw_pagamentos, patient_id)
    data["pagamentos"] = pagamentos_filtrados
    if debug:
        data.setdefault("_debug", {})["exceptions_applied"] = ignores_aplicados

    # Financeiro
    total = sum(p.get("valor", 0) for p in propostas)
    pago = sum(p.get("valor", 0) for p in data["pagamentos"])
    saldo = max(total - pago, 0.0)
    data["financeiro"] = {"total": total, "pago": pago, "saldo": saldo}

    if debug:
        data["_debug"] = {"qtd_propostas": len(propostas), "total_propostas": total}
        dbg = data.setdefault("_debug", {})
        dbg["qtd_propostas"] = len(propostas)
        dbg["total_propostas"] = total

    # ======= AGENDAMENTOS (reais, janela curta por padrão) =======
    # ======= AGENDAMENTOS (ano inteiro costurado em janelas <= 180d) =======
    try:
        today = date.today()
        start = date(today.year, 1, 1)
        end   = date(today.year, 12, 31)
        stitched = feegow.get_appointments_range_hydrated(patient_id, start, end)
        data["agendamentos"] = stitched["items"]  # já vem com nomes traduzidos
        if debug:
            data.setdefault("_debug", {})["windows"] = stitched.get("windows", [])
    except httpx.HTTPError:
        data["agendamentos"] = []

    if debug:
        data.setdefault("_debug", {})["qtd_agendamentos"] = len(data.get("agendamentos", []))

    return data

# ------------------------------------------------------------------
# Rotas de depuração
# ------------------------------------------------------------------
@app.get("/_debug/config")
def dbg_config():
    token = os.getenv("FEEGOW_TOKEN", "")
    return {
        "base": os.getenv("FEEGOW_BASE"),
        "auth_header": os.getenv("FEEGOW_AUTH_HEADER"),
        "auth_scheme": os.getenv("FEEGOW_AUTH_SCHEME"),
        "token_present": bool(token),
        "token_len": len(token or ""),
    }

@app.get("/_debug/feegow/appoints/{patient_id}")
def dbg_appoints(
    patient_id: int,
    before: int = Query(90, ge=0, le=180),   # dias para trás
    after:  int = Query(90, ge=0, le=180),   # dias para frente
):
    return feegow.debug_appoints(patient_id, days_before=before, days_after=after)

@app.get("/_debug/feegow/appoints_range/{patient_id}")
def dbg_appoints_range(
    patient_id: int,
    start: str = Query(..., description="Data inicial ISO: AAAA-MM-DD"),
    end:   str = Query(..., description="Data final ISO: AAAA-MM-DD"),
    hydrate: int = Query(1, description="1 = trazer nomes traduzidos"),
):
    try:
        d_start = datetime.strptime(start, "%Y-%m-%d").date()
        d_end   = datetime.strptime(end,   "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Datas devem estar no formato AAAA-MM-DD")

    if hydrate:
        return feegow.get_appointments_range_hydrated(patient_id, d_start, d_end)
    return feegow.get_appointments_range(patient_id, d_start, d_end)

@app.get("/debug/feegow/list-invoice")
def _debug_list_invoice(invoice_id: str,
                        data_start: str = "01-01-2000",
                        data_end: str   = "31-08-2050",
                        tipo_transacao: str = "C"):
    return feegow.debug_list_invoice(invoice_id, data_start, data_end, tipo_transacao)


@app.get("/health")
def health():
    return {"status": "ok"}

