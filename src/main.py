#!/usr/bin/env python3
"""
Menu principal do sistema de recomendação de filmes.
Permite escolher qual funcionalidade executar.
"""

import sys
import subprocess
from pathlib import Path

# Cores para o terminal (opcional, funciona sem cores também)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header():
    """Imprime o cabeçalho do menu."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print("  SISTEMA DE RECOMENDAÇÃO DE FILMES - MENU PRINCIPAL")
    print(f"{'='*60}{Colors.END}\n")


def print_menu():
    """Imprime as opções do menu."""
    print(f"{Colors.CYAN}Opções disponíveis:{Colors.END}\n")

    print(f"{Colors.BOLD}1.{Colors.END} Preparar dados ReDial")
    print(f"   {Colors.YELLOW}→ prepare_redial_from_jsonl.py{Colors.END}")
    print(f"   Processar dados do formato JSONL\n")
    
    print(f"{Colors.BOLD}2.{Colors.END} Testar carregamento de dados")
    print(f"   {Colors.YELLOW}→ test_redial_loader.py{Colors.END}")
    print(f"   Verificar se o dataset carrega corretamente\n")
    
    print(f"{Colors.BOLD}3.{Colors.END} Treinamento ReDial (dataset completo)")
    print(f"   {Colors.YELLOW}→ train_redial_full.py{Colors.END}")
    print(f"   Treinamento completo com todos os dados\n")
    
    print(f"{Colors.BOLD}4.{Colors.END} Inferência/Predição (ReDial)")
    print(f"   {Colors.YELLOW}→ infer_redial_single.py{Colors.END}")
    print(f"   Fazer predições de filmes para uma conversa\n")
    
    print(f"{Colors.BOLD}5.{Colors.END} Avaliação de métricas (ReDial)")
    print(f"   {Colors.YELLOW}→ eval_redial_metrics.py{Colors.END}")
    print(f"   Avaliar modelo com métricas de recomendação\n")
    
    print(f"{Colors.BOLD}6.{Colors.END} Avaliação de dataset (ReDial)")
    print(f"   {Colors.YELLOW}→ redial_dataset_eval.py{Colors.END}")
    print(f"   Analisar estatísticas do dataset\n")

    print("-"*60)
    print("Opções para MovieLens")
    print("-"*60)

    print(f"{Colors.BOLD}7.{Colors.END} Preparar dados MovieLens")
    print(f"   {Colors.YELLOW}→ preprocess_movielens_tags.py{Colors.END}")
    print(f"   Processar dados do formato JSONL\n")

    print(f"{Colors.BOLD}8.{Colors.END} Treinamento Multi-Task (ReDial + MovieLens)")
    print(f"   {Colors.YELLOW}→ train_multitask.py{Colors.END}")
    print(f"   Treinamento multi-task com ReDial e MovieLens\n")

    print(f"{Colors.BOLD}9.{Colors.END} Avaliação Multi-Task (ReDial + MovieLens)")
    print(f"   {Colors.YELLOW}→ eval_redial_multitask.py{Colors.END}")
    print(f"   Avaliação multi-task com ReDial e MovieLens\n")

    
    print(f"{Colors.BOLD}0.{Colors.END} Sair\n")


def get_user_choice():
    """Solicita e retorna a escolha do usuário."""
    while True:
        try:
            choice = input(f"{Colors.GREEN}Escolha uma opção (1-9): {Colors.END}").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return choice
            else:
                print(f"{Colors.RED}Opção inválida! Digite um número entre 1 e 9.{Colors.END}\n")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Operação cancelada pelo usuário.{Colors.END}")
            sys.exit(0)
        except EOFError:
            print(f"\n{Colors.YELLOW}Saindo...{Colors.END}")
            sys.exit(0)


def execute_script(script_name):
    """Executa um script Python."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"{Colors.RED}Erro: Arquivo '{script_name}' não encontrado!{Colors.END}")
        return False
    
    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"Executando: {script_name}")
    print(f"{'='*60}{Colors.END}\n")
    
    try:
        # Executa o script usando o mesmo interpretador Python
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=False
        )
        
        print(f"\n{Colors.CYAN}{'='*60}")
        if result.returncode == 0:
            print(f"{Colors.GREEN}Script executado com sucesso!{Colors.END}")
        else:
            print(f"{Colors.RED}Script terminou com código de erro: {result.returncode}{Colors.END}")
        print(f"{'='*60}{Colors.END}\n")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Execução interrompida pelo usuário.{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Erro ao executar script: {e}{Colors.END}")
        return False


def main():
    """Função principal do menu."""
    # Mapeamento de opções para scripts
    scripts = {
        '1': 'prepare_redial_from_jsonl.py',
        '2': 'test_redial_loader.py',
        '3': 'train_redial_full.py',
        '4': 'infer_redial_single.py',
        '5': 'eval_redial_metrics.py',
        '6': 'redial_dataset_eval.py',
        '7': 'preprocess_movielens_tags.py',
        '8': 'train_multitask.py',
        '9': 'eval_redial_multitask.py',
    }
    
    while True:
        print_header()
        print_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print(f"\n{Colors.GREEN}Obrigado por usar o sistema! Até logo!{Colors.END}\n")
            break
        
        script_name = scripts.get(choice)
        if script_name:
            execute_script(script_name)
            
            # Pergunta se deseja continuar
            if choice != '0':
                try:
                    continue_choice = input(
                        f"\n{Colors.YELLOW}Deseja executar outra funcionalidade? (s/n): {Colors.END}"
                    ).strip().lower()
                    if continue_choice not in ['s', 'sim', 'y', 'yes']:
                        print(f"\n{Colors.GREEN}Até logo!{Colors.END}\n")
                        break
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{Colors.GREEN}Até logo!{Colors.END}\n")
                    break


if __name__ == "__main__":
    main()

