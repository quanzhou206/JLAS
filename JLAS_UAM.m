clc;
clear;
warning off;

range = 400;
Height = 200;
t_all = 6;% 8 An at least 4
dt = 1;
A_all = 8;
alpha = 10^(-6);
beta = 10^(-8);
c = 299792458;
L_times = 1000;

clock_offset = (2*rand(A_all,1)-1)*alpha;
clock_skew = (2*rand(A_all,1)-1)*beta;

Anchors = [0 0 0;
           range 0 0;
           0 range 0;
           0 0 Height;
           range range 0;
           range 0 Height;
           0 range Height;
           range range Height;]';

x0 = (2*rand(3,1)-1).*[100;100;50]+[200;200;100];
v = (2*rand(3,1)-1);
v = 25*v/norm(v);
a = (2*rand(3,1)-1);
a = 5*a/norm(a);

pos = zeros(3,t_all);
pos(:,1) = x0;

for t = 1:t_all
    pos(:,t) = x0 + v*(t-1)*dt + a*((t-1)*dt)^2/2;
end


sig_all = [10^(-3),10^(-2.5),10^(-2),10^(-1.5),10^(-1),10^(-0.5),10^(0)];

err_sig_x = zeros(length(sig_all),1);
err_sig_v = zeros(length(sig_all),1);
err_sig_a = zeros(length(sig_all),1);
err_sig_alpha = zeros(length(sig_all),1);
err_sig_beta = zeros(length(sig_all),1);
err_sig_x_CRLB = zeros(length(sig_all),1);
err_sig_v_CRLB = zeros(length(sig_all),1);
err_sig_a_CRLB = zeros(length(sig_all),1);
err_sig_alpha_CRLB = zeros(length(sig_all),1);
err_sig_beta_CRLB = zeros(length(sig_all),1);

for sig_now = 1:length(sig_all)
    errx = zeros(L_times,1);
    errv = zeros(L_times,1);
    erra = zeros(L_times,1);
    err_alpha = zeros(L_times,1);
    err_beta = zeros(L_times,1);
    flag_all = zeros(L_times,1);
    for manto = 1:L_times
        %%测距
        range = zeros(A_all,t_all);
        for t=1:t_all
            for an = 1:A_all
                range(an,t) = norm(Anchors(:,an)-pos(:,t)) + c*(clock_offset(an)+clock_skew(an)*(t-1))+normrnd(0,sig_all(sig_now));
            end
        end
        
        theta_hat = [x0;zeros(3,1);zeros(3,1);clock_offset*c;clock_skew*c]; % x0 v a dt df
        dtheta = 100*ones(length(theta_hat),1);
        iter = 1;flag = 1;
        tic
        while(norm(dtheta)>1e-3)
            G_all = [];
            b_all = [];
            for local_t = 1:t_all
                [G,b] = Jacobi(theta_hat,Anchors,range(:,local_t),(local_t-1)*dt);
                G_all = [G_all;G];
                b_all = [b_all;b];
            end
            dtheta = (G_all'*G_all)\G_all'*b_all;
            theta_hat = theta_hat + dtheta;
            iter = iter+1;
            if iter > 20
                flag = 0;
                break;
            end
        end
        flag_all(manto) = flag;
        time(manto) = toc;
        if flag == 1
            if (sum(isnan(theta_hat))==0)
                errx(manto,1) = norm(theta_hat(1:3)-x0)^2;
                errv(manto,1) = norm(theta_hat(4:6)-v)^2;
                erra(manto,1) = norm(theta_hat(7:9)-a)^2;
                err_alpha(manto,1) = norm(theta_hat(10:17)/c-clock_offset)^2;
                err_beta(manto,1) = norm(theta_hat(18:end)/c-clock_skew)^2;
            else
                flag_all(manto) = 0;
            end
        end
    end
    [sigx,sigv,siga,sigalpha,sigbeta] = CRLB([x0;v;a],Anchors,sig_all(sig_now),t_all,dt);
    err_sig_x(sig_now) = sqrt(sum(errx)/sum(flag_all));
    err_sig_v(sig_now) = sqrt(sum(errv)/sum(flag_all));
    err_sig_a(sig_now) = sqrt(sum(erra)/sum(flag_all));
    err_sig_alpha(sig_now) = sqrt(sum(err_alpha)/sum(flag_all)/A_all);
    err_sig_beta(sig_now) = sqrt(sum(err_beta)/sum(flag_all)/A_all);

    err_sig_x_CRLB(sig_now) = sqrt(sigx);
    err_sig_v_CRLB(sig_now) = sqrt(sigv);
    err_sig_a_CRLB(sig_now) = sqrt(siga);
    err_sig_alpha_CRLB(sig_now) = sqrt(sigalpha/A_all)/c;
    err_sig_beta_CRLB(sig_now) = sqrt(sigbeta/A_all)/c;
end

figure;
subplot(3,1,1);
loglog(sig_all,err_sig_x,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_x_CRLB,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-UAM (p)','CRLB (p)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE (p) (m)','FontSize',10,'interpreter','latex');

subplot(3,1,2);
loglog(sig_all,err_sig_alpha*c,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_alpha_CRLB*c,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-UAM ($\alpha$)','CRLB ($\alpha$)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE ($\alpha$) (m)','FontSize',10,'interpreter','latex');

subplot(3,1,3);
loglog(sig_all,err_sig_beta*c,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_beta_CRLB*c,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-UAM ($\beta$)','CRLB ($\beta$)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE ($\beta$) (m/s)','FontSize',10,'interpreter','latex');
% ax=gcf;
% ax.Position = [2525.8,40.2,560,450];


figure;
subplot(2,1,1);
loglog(sig_all,err_sig_v,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_v_CRLB,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-UAM (v)','CRLB (v)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE (v) (m/s)','FontSize',10,'interpreter','latex');axis tight

subplot(2,1,2);
loglog(sig_all,err_sig_a,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_a_CRLB,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-UAM (a)','CRLB (a)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE (a) (${\rm m/s^2}$)','FontSize',10,'interpreter','latex');axis tight
% ax=gcf;
% ax.Position = [2525.8,40.2,560,450];

function [G,b] = Jacobi(theta,anchors,range,delta)
    G = zeros(length(range),length(theta));
    b = zeros(length(range),1);
    x = theta(1:3);
    v = theta(4:6);
    a = theta(7:9);
    for an = 1:length(range)
        t0 = anchors(:,an) - x - v*delta - a*delta^2/2;
        G(an,1:3) = - t0/norm(t0);
        G(an,4:6) = - delta*t0/norm(t0);
        G(an,7:9) = t0/norm(t0) - delta^2*t0/norm(t0)/2;
        G(an,9+an) = 1;
        G(an,9+an+length(range)) = delta;
        b(an) = range(an) - (norm(anchors(:,an) - x - v*delta - a*delta^2/2)+theta(9+an)+delta*theta(9+an+length(range)));
    end
end

function [sigx,sigv,siga,sigalpha,sigbeta] = CRLB(theta,anchors,sig,t_all,dt)
    x = theta(1:3);
    v = theta(4:6);
    a = theta(7:9);
    G_all = [];
    for local_t = 1:t_all
        delta = (local_t-1)*dt;
        G = zeros(size(anchors,2),length(theta));
        for an = 1:size(anchors,2)
            t0 = anchors(:,an) - x - v*delta - a*delta^2/2;
            G(an,1:3) = - t0/norm(t0);
            G(an,4:6) = - delta*t0/norm(t0);
            G(an,7:9) = t0/norm(t0) - delta^2*t0/norm(t0)/2;
            G(an,9+an) = 1;
            G(an,9+an+size(anchors,2)) = delta;
        end
        G_all = [G_all;G];
    end
    Fim = inv(G_all'*G_all/(sig^2));
    sigx = trace(Fim(1:3,1:3));
    sigv = trace(Fim(4:6,4:6));
    siga = trace(Fim(7:9,7:9));
    sigalpha = trace(Fim(10:17,10:17));
    sigbeta = trace(Fim(18:end,18:end));
end