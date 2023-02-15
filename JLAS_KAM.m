clc;
clear;
warning off;

range = 400;
Height = 200;
t_all = 6;% 8 An at least 3
A_all = 8;
alpha = 10^(-6);
% alpha = 5;
beta = 10^(-8);
mode = 2; %  1 constant velocity; 2 acceleration

c = 299792458;
L_times = 2000;

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

x0 = (2*rand(3,1)-1).*[10;10;10]+[200;200;300];

% x0 = Anchors(:,4)+10;

if mode==1
    v = (2*rand(3,1)-1);
    v = 15*v/norm(v);
    a = zeros(3,1);
elseif mode==2
    v = (2*rand(3,1)-1);
    v = 15*v/norm(v);
    a = (2*rand(3,1)-1);
    a = 5*a/norm(a);
else
    v = zeros(3,1);
    a = zeros(3,1);
end

pos = zeros(3,t_all);
pos(:,1) = x0;

for t = 1:t_all
    pos(:,t) = x0 + v*(t-1) + a*(t-1)^2/2;
end

% load trajectory.mat

% figure;
% scatter3(Anchors(1,:),Anchors(2,:),Anchors(3,:),100,'p',...
%         'MarkerEdgeColor',[0.8500 0.3250 0.0980],...
%         'MarkerFaceColor',[0.8500 0.3250 0.0980]);grid on;hold on;
% plot3(pos(1,:),pos(2,:),pos(3,:));


sig_all = [10^(-3),10^(-2.5),10^(-2),10^(-1.5),10^(-1),10^(-0.5),10^(0)];

err_sig = zeros(length(sig_all),1);
err_sig_alpha = zeros(length(sig_all),1);
err_sig_beta = zeros(length(sig_all),1);
err_sig_x_CRLB = zeros(length(sig_all),1);
err_sig_alpha_CRLB = zeros(length(sig_all),1);
err_sig_beta_CRLB = zeros(length(sig_all),1);

for sig_now = 1:length(sig_all)
    err = zeros(L_times,1);
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
        
        tic
        theta_hat = [x0;zeros(2*A_all,1)];
        dtheta = 100*ones(length(theta_hat),1);
        iter = 1;flag = 1;
        while(norm(dtheta)>1e-3)
            G_all = [];
            b_all = [];
            for local_t = 1:t_all
                [G,b] = Jacobi(theta_hat,Anchors,range(:,local_t),local_t-1,a,v);
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
        time(manto) = toc;
        flag_all(manto) = flag;
        if flag == 1
            err(manto,1) = norm(theta_hat(1:3)-x0)^2;
            err_alpha(manto,1) = norm(theta_hat(4:11)/c-clock_offset)^2;
            err_beta(manto,1) = norm(theta_hat(12:end)/c-clock_skew)^2;
        end
    end
    err_sig(sig_now) = sqrt(sum(err)/sum(flag_all));
    err_sig_alpha(sig_now) = sqrt(sum(err_alpha)/sum(flag_all)/A_all);
    err_sig_beta(sig_now) = sqrt(sum(err_beta)/sum(flag_all)/A_all);
    [sigx,sigalpha,sigbeta] = CRLB(x0,Anchors,a,v,t_all,sig_all(sig_now));
    err_sig_x_CRLB(sig_now) = sqrt(sigx);
    err_sig_alpha_CRLB(sig_now) = sqrt(sigalpha/A_all)/c;
    err_sig_beta_CRLB(sig_now) = sqrt(sigbeta/A_all)/c;
end

figure;
subplot(3,1,1);
loglog(sig_all,err_sig,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_x_CRLB,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-KAM (p)','CRLB (p)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE (p) (m)','FontSize',10,'interpreter','latex');

subplot(3,1,2);
loglog(sig_all,err_sig_alpha*c,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_alpha_CRLB*c,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-KAM ($\alpha$)','CRLB ($\alpha$)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE ($\alpha$) (m)','FontSize',10,'interpreter','latex');

subplot(3,1,3);
loglog(sig_all,err_sig_beta*c,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_beta_CRLB*c,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-KAM ($\beta$)','CRLB ($\beta$)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma (m)$','FontSize',10,'interpreter','latex');
ylabel('RMSE ($\beta$) (m/s)','FontSize',10,'interpreter','latex');
ax=gcf;
ax.Position = [2525.8,40.2,560,450];


function [G,b] = Jacobi(theta,anchors,range,delta,a,v)
    G = zeros(length(range),length(theta));
    b = zeros(length(range),1);
    for an = 1:length(range)
        t0 = anchors(:,an) - theta(1:3) - v*delta - a*delta^2/2;
        G(an,1:3) = - t0/norm(t0);
        G(an,3+an) = 1;
        G(an,3+an+length(range)) = delta;
        b(an) = range(an) - (norm(anchors(:,an) - theta(1:3) - v*delta - a*delta^2/2)+theta(3+an)+delta*theta(3+an+length(range)));
    end
end

function [sigx,sigalpha,sigbeta] = CRLB(theta,anchors,a,v,t_all,sig)
    G_all = [];
    for local_t = 1:t_all
        delta = local_t-1;
        G = zeros(size(anchors,2),length(theta));
        for an = 1:size(anchors,2)
            t0 = anchors(:,an) - theta(1:3) - v*delta - a*delta^2/2;
            G(an,1:3) = - t0/norm(t0);
            G(an,3+an) = 1;
            G(an,3+an+size(anchors,2)) = delta;
        end
        G_all = [G_all;G];
    end
    Fim = inv(G_all'*G_all/(sig^2));
    sigx = trace(Fim(1:3,1:3));
    sigalpha = trace(Fim(4:11,4:11));
    sigbeta = trace(Fim(12:end,12:end));
end